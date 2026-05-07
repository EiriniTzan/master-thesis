import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from tscls.core.results import StepResult
from tscls.core.sample import Sample
from tscls.detection.ae_detector import AEDetector
from tscls.detection.threshold import ThresholdRule
from tscls.models.dnn_classifier import DNNClassifier
from tscls.models.stream_dnn import StreamDNN
from tscls.models.autoencoder import Autoencoder
from tscls.pipeline.pipeline_config import PipelineConfig
from tscls.training.ae_trainer import AETrainer
from tscls.training.base_trainer import BaseTrainer
from tscls.training.stream_trainer import StreamTrainer


class Detector:
    """
    Encapsulates offline training and per-sample online detection.

    Offline phase (fit):
      1. Fit StandardScaler on reference data.
      2. Train the base DNN (Model 1) as a classifier.
      3. Extract latent representations from the last hidden layer.
      4. Train an Autoencoder on those latents.
      5. Calibrate a k-sigma threshold on reference reconstruction errors.
      6. Clone the base DNN into a Stream DNN (Model 2), freezing early layers.

    Online phase (detect):
      For each incoming labelled sample the method scales the input, runs a
      forward pass through the Stream DNN, computes the AE reconstruction
      error, checks the threshold, performs one gradient update on the
      trainable layers, and returns a StepResult.

    Note: detect() requires a label because online adaptation of the Stream
    DNN is an integral part of each step (per the paper's methodology).
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Parameters
        ----------
        config : PipelineConfig
            Full pipeline configuration.
        """

        self.config = config
        self.device = (
            config.device
            if config.device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.scaler = StandardScaler()

        self.model1 = DNNClassifier(
            layer_sizes=config.model.layer_sizes,
            activation=config.model.activation,
        )

        self.model3 = Autoencoder(
            encoder_sizes=config.autoencoder.encoder_sizes,
            activation=config.autoencoder.activation,
        )

        self.threshold_rule = ThresholdRule(k=config.autoencoder.threshold_k)

        self._ae_detector = AEDetector(
            autoencoder=self.model3,
            threshold_rule=self.threshold_rule,
            device=self.device,
        )

        self._base_trainer = BaseTrainer(
            loss_function=self._make_loss(config.optimization.model1_loss_name),
            gamma1=config.optimization.gamma1,
            s1=config.optimization.s1,
            device=self.device,
        )

        self._ae_trainer = AETrainer(
            loss_function=self._make_loss(config.optimization.model3_loss_name),
            device=self.device,
        )

        self._stream_trainer = StreamTrainer(
            loss_function=self._make_loss(config.optimization.model2_loss_name),
            gamma2=config.optimization.gamma2,
            s2=config.optimization.s2,
            device=self.device,
        )

        self.model2: StreamDNN | None = None
        self._stream_optimizer: torch.optim.Optimizer | None = None

        self.model1_train_losses: list[float] = []
        self.model3_train_losses: list[float] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_loss(self, name: str) -> nn.Module:
        if name == "bce_with_logits":
            return nn.BCEWithLogitsLoss()
        if name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unknown loss function: {name!r}")

    def _make_optimizer(
        self,
        name: str,
        params,
        lr: float,
    ) -> torch.optim.Optimizer:
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr)
        if name == "adam":
            return torch.optim.Adam(params, lr=lr)
        raise ValueError(f"Unknown optimizer: {name!r}")

    # ------------------------------------------------------------------
    # Offline phase
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Run the offline training phase on reference (pre-drift) data.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor of shape (n, input_dim)
            Reference feature matrix.
        y : np.ndarray or torch.Tensor of shape (n,) or (n, 1)
            Reference labels.
        """

        # Convert inputs to numpy / torch as required downstream
        x_np = (
            X if isinstance(X, np.ndarray)
            else X.detach().cpu().numpy()
        )
        y = (
            torch.tensor(np.asarray(y, dtype=np.float32))
            if isinstance(y, np.ndarray)
            else y
        )
        x_scaled = torch.tensor(
            self.scaler.fit_transform(x_np), dtype=torch.float32
        )

        # Train Model 1 (reference DNN)
        model1_optimizer = self._make_optimizer(
            self.config.optimization.model1_optimizer_name,
            self.model1.parameters(),
            self.config.optimization.model1_learning_rate,
        )
        self.model1_train_losses = self._base_trainer.train_model(
            model=self.model1,
            x_reference=x_scaled,
            y_reference=y,
            optimizer=model1_optimizer,
            epochs=self.config.training.model1_epochs,
            batch_size=self.config.training.batch_size,
        )

        # Extract latent representations
        latents = self._base_trainer.extract_latents(
            model=self.model1,
            x_reference=x_scaled,
        )
        latents = latents[-5000:]
        
        # Train Model 3 (autoencoder)
        model3_optimizer = self._make_optimizer(
            self.config.optimization.model3_optimizer_name,
            self.model3.parameters(),
            self.config.optimization.model3_learning_rate,
        )
        self.model3_train_losses = self._ae_trainer.train_model(
            autoencoder=self.model3,
            reference_latents=latents,
            optimizer=model3_optimizer,
            epochs=self.config.training.model3_epochs,
            batch_size=self.config.training.batch_size,
        )

        # Calibrate threshold on reference reconstruction errors
        self.model3.to(self.device).eval()
        with torch.no_grad():
            _, ref_errors = self.model3.reconstruction_error(
                latents.to(self.device)
            )
        self.threshold_rule.calibrate(ref_errors.cpu())

        # Initialise Model 2 (stream DNN)
        self.model2 = StreamDNN(base_model=self.model1)
        self._stream_optimizer = self._make_optimizer(
            self.config.optimization.model2_optimizer_name,
            self.model2.trainable_parameters(),
            self.config.optimization.model2_learning_rate,
        )

    # ------------------------------------------------------------------
    # Online phase
    # ------------------------------------------------------------------

    def detect(self, sample: Sample) -> StepResult:
        """
        Process one labelled stream sample: adapt the Stream DNN and
        check for concept drift.

        Parameters
        ----------
        sample : Sample
            A single stream sample. sample.y must not be None because
            online adaptation requires a label.

        Returns
        -------
        StepResult
            Prediction, latent vector, reconstruction error, drift flag,
            and training loss for this sample.

        Raises
        ------
        RuntimeError
            If fit() has not been called yet.
        """

        if self.model2 is None or self._stream_optimizer is None:
            raise RuntimeError(
                "Detector.fit() must be called before detect()."
            )

        # Scale input
        x_scaled = self.scaler.transform(
            np.array(sample.x, dtype=np.float32).reshape(1, -1)
        )[0].astype(np.float32)

        scaled_sample = Sample(x=x_scaled, y=sample.y, index=sample.index)

        # Online forward + adaptation step
        loss, logits, latent = self._stream_trainer.train_on_sample(
            model=self.model2,
            sample=scaled_sample,
            optimizer=self._stream_optimizer,
        )

        # Drift detection via AE reconstruction error
        detection = self._ae_detector.detect(latent)
        prediction = float(torch.sigmoid(logits).item())

        return StepResult(
            sample_index=sample.index,
            prediction=prediction,
            latent_vector=latent.squeeze(0).cpu().numpy(),
            reconstruction_error=detection.reconstruction_error,
            drift_detected=detection.drift_detected,
            training_loss=loss,
        )
