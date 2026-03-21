import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

from tscls.core.results import StepResult, PipelineResult
from tscls.core.sample import Sample
from tscls.detection.ae_detector import DriftDetector
from tscls.detection.threshold import ThresholdRule
from tscls.models.autoencoder import Autoencoder
from tscls.models.base_dnn import DNNBase
from tscls.models.stream_dnn import StreamDNN
from tscls.pipeline.pipeline_config import PipelineConfig
from tscls.simulation.stream_simulation import StreamSimulator
from tscls.training.ae_trainer import AETrainer
from tscls.training.base_trainer import BaseTrainer
from tscls.training.stream_trainer import StreamTrainer


class DNNAEDDPipeline:
    """
    Full pipeline of the DNN + AE drift detection method.
    This pipeline implements:
    1) reference data scaling
    2) reference model 1 training
    3) latent extraction
    4) autoencoder training
    5) threshold calibration
    6) streaming model 2 initialization
    7) online model 2 adaptation and drift detection
    """

    def __init__(self, config: PipelineConfig) -> None:
        """
        Initialize the DNN + AE drift detection pipeline.

        Parameters
        ----------
        config : PipelineConfig
            The configuration of the pipeline.

        Attributes
        ----------
        config : PipelineConfig
            The configuration of the pipeline.
        device : str
            The device on which computations should be performed.
        scaler : StandardScaler
            The scaler used to standardize the input data.
        base_model : DNNBase
            The reference DNN model.
        autoencoder : Autoencoder
            The autoencoder model used for reconstruction error calculation.
        threshold_rule : ThresholdRule
            The thresholding rule used to decide whether a sample is drifted or not.
        detector : DriftDetector
            The drift detector used to detect drift.
        base_trainer : BaseTrainer
            The trainer used to train the reference DNN model.
        ae_trainer : AETrainer
            The trainer used to train the autoencoder model.
        stream_trainer : StreamTrainer
            The trainer used to train the adaptive stream model.
        stream_model : StreamDNN | None
            The adaptive stream model.
        """

        self.config = config
        self.device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()

        self.base_model = DNNBase(
            input_dim=config.model.input_dim,
            hidden_dims=config.model.hidden_dims,
            output_dim=config.model.output_dim,
        )

        self.autoencoder = Autoencoder(
            input_dim=config.autoencoder.input_dim,
            hidden_dim=config.autoencoder.hidden_dim,
        )

        self.threshold_rule = ThresholdRule(
            k=config.autoencoder.threshold_k
        )

        self.detector = DriftDetector(
            autoencoder=self.autoencoder,
            threshold_rule=self.threshold_rule,
            device=self.device,
        )

        self.base_trainer = BaseTrainer(
            loss_function=self._get_loss(config.optimization.base_loss_name),
            gamma1=config.optimization.gamma1,
            s1=config.optimization.s1,
            device=self.device,
        )

        self.ae_trainer = AETrainer(
            loss_function=self._get_loss(config.optimization.ae_loss_name),
            device=self.device,
        )

        self.stream_trainer = StreamTrainer(
            loss_function=self._get_loss(config.optimization.stream_loss_name),
            gamma2=config.optimization.gamma2,
            s2=config.optimization.s2,
            device=self.device,
        )

        self.stream_model: StreamDNN | None = None
        self.base_train_losses: list[float] = []
        self.ae_train_losses: list[float] = []

    def _get_loss(self, name: str) -> nn.Module:
        """
        Get a loss function object based on the given name.

        Parameters
        ----------
        name : str
            The name of the loss function.

        Returns
        -------
        nn.Module
            The loss function object.

        Supported loss functions are:
        - "bce_with_logits" (nn.BCEWithLogitsLoss)
        - "mse" (nn.MSELoss)
        """
        
        if name == "bce_with_logits":
            return nn.BCEWithLogitsLoss()
        if name == "mse":
            return nn.MSELoss()

    def _get_optimizer(self, name: str, params, lr: float) -> torch.optim.Optimizer:
        """
        Get an optimizer object based on the given name, parameters, and learning rate.

        Parameters
        ----------
        name : str
            The name of the optimizer.
        params : iterable
            The parameters of the model to be optimized.
        lr : float
            The learning rate of the optimizer.

        Returns
        -------
        torch.optim.Optimizer
            The optimizer object.

        Supported optimizers are:
        - "sgd" (torch.optim.SGD)
        - "adam" (torch.optim.Adam)
        """

        if name == "sgd":
            return torch.optim.SGD(params, lr=lr)
        if name == "adam":
            return torch.optim.Adam(params, lr=lr)

    # OFFLINE PHASE

    def _fit_reference_scaler(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fit the reference scaler to the input data and scale it.

        Parameters
        ----------
        x : torch.Tensor
            The input data to be scaled.

        Returns
        -------
        torch.Tensor
            The scaled input data.
        """
        
        x_np = x.detach().cpu().numpy()
        x_scaled = self.scaler.fit_transform(x_np)
        return torch.tensor(x_scaled, dtype=torch.float32)

    def _transform_stream_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Transform a stream sample using the reference scaler.

        Parameters
        ----------
        x : np.ndarray
            The stream sample to be transformed.

        Returns
        -------
        np.ndarray
            The transformed stream sample.
        """
        
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        return x_scaled[0].astype(np.float32)

    def _initialize_stream_model(self) -> None:
        """
        Initialize the StreamDNN model.

        The StreamDNN model is initialized based on the pre-trained
        reference DNN model and the freeze_before_layer hyperparameter.
        The StreamDNN model is then used for online drift detection.
        """
        
        self.stream_model = StreamDNN(
            base_model=self.base_model,
            freeze_before_layer=self.config.stream.freeze_before_layer,
            )
        

    def _calibrate_threshold(self, latents: torch.Tensor) -> None:
        """
        Calibrate the thresholding rule using the reconstruction errors of the
        autoencoder on the reference latent representations.

        Parameters
        ----------
        latents : torch.Tensor
            The latent representations of the reference data points.
        """

        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        latents = latents.to(self.device)

        with torch.no_grad():
            _, errors = self.autoencoder.reconstruction_error(latents)

        self.threshold_rule.calibrate(errors.cpu())

    def prepare_offline_phase(
        self,
        x_reference: torch.Tensor,
        y_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prepare the offline phase by fitting the reference scaler to the input data,
        training the base DNN model on the scaled input data, extracting the latent
        representations, training the autoencoder on the latent representations,
        calibrating the thresholding rule, and initializing the StreamDNN model.

        Parameters
        ----------
        x_reference : torch.Tensor
            The input data points to be used for training the base DNN model.
        y_reference : torch.Tensor
            The target data points to be used for training the base DNN model.

        Returns
        -------
        torch.Tensor
            The latent representations of the reference data points.
        """

        x_scaled = self._fit_reference_scaler(x_reference)

        base_optimizer = self._get_optimizer(
            self.config.optimization.base_optimizer_name,
            self.base_model.parameters(),
            self.config.optimization.base_learning_rate,
        )

        self.base_train_losses = self.base_trainer.train_model(
            model=self.base_model,
            x_reference=x_scaled,
            y_reference=y_reference,
            optimizer=base_optimizer,
            epochs=self.config.training.base_epochs,
        )

        latents = self.base_trainer.extract_latents(
            model=self.base_model,
            x_reference=x_scaled,
        )

        ae_optimizer = self._get_optimizer(
            self.config.optimization.ae_optimizer_name,
            self.autoencoder.parameters(),
            self.config.optimization.ae_learning_rate,
        )

        self.ae_train_losses = self.ae_trainer.train_model(
            autoencoder=self.autoencoder,
            reference_latents=latents,
            optimizer=ae_optimizer,
            epochs=self.config.training.ae_epochs,
        )

        self._calibrate_threshold(latents)
        self._initialize_stream_model()

        return latents

    # ONLINE PHASE

    def online_step(
        self,
        sample: Sample,
        optimizer: torch.optim.Optimizer,
    ) -> StepResult:
        """
        Perform an online supervised update of the StreamDNN model on a single sample.

        Parameters
        ----------
        sample : Sample
            The sample to be used for updating the model.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for updating the model.

        Returns
        -------
        StepResult
            A StepResult object containing the result of the online step.
        """
        
        scaled_sample = Sample(
            x=self._transform_stream_sample(sample.x),
            y=sample.y,
            index=sample.index,
        )

        loss, logits, latent = self.stream_trainer.train_on_sample(
            model=self.stream_model,
            sample=scaled_sample,
            optimizer=optimizer,
        )

        detection = self.detector.detect(latent)
        prediction = float(torch.sigmoid(logits).item())

        return StepResult(
            sample_index=sample.index,
            prediction=prediction,
            latent_vector=latent.squeeze(0).cpu().numpy(),
            reconstruction_error=detection.reconstruction_error,
            drift_detected=detection.drift_detected,
            training_loss=loss,
        )

    def run_full_pipeline(
        self,
        x_reference: torch.Tensor,
        y_reference: torch.Tensor,
        stream: StreamSimulator,
    ) -> PipelineResult:
        """
        Run the full pipeline of drift detection, including the offline phase and
        the online phase.

        Parameters
        ----------
        x_reference : torch.Tensor
            The input data points to be used for training the base DNN model.
        y_reference : torch.Tensor
            The target data points to be used for training the base DNN model.
        stream : StreamSimulator
            The stream simulator object.

        Returns
        -------
        PipelineResult
            A PipelineResult object containing the results of the pipeline.
        """
        
        self.prepare_offline_phase(x_reference, y_reference)

        stream_optimizer = self._get_optimizer(
            self.config.optimization.stream_optimizer_name,
            self.stream_model.trainable_parameters(),
            self.config.optimization.stream_learning_rate,
        )

        stream.reset()

        step_results: list[StepResult] = []
        drift_points: list[int] = []
        reconstruction_errors: list[float] = []
        predictions: list[float] = []
        training_losses: list[float] = []

        while stream.has_next_sample():
            x, y, i = stream.next_sample()

            sample = Sample(
                x=x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x,
                y=int(y.item()) if isinstance(y, torch.Tensor) else y,
                index=i,
            )

            step_result = self.online_step(
                sample=sample,
                optimizer=stream_optimizer,
            )

            step_results.append(step_result)
            reconstruction_errors.append(step_result.reconstruction_error)
            predictions.append(step_result.prediction)
            training_losses.append(step_result.training_loss)

            if step_result.drift_detected:
                drift_points.append(step_result.sample_index)

        return PipelineResult(
            step_results=step_results,
            drift_points=drift_points,
            reconstruction_errors=reconstruction_errors,
            predictions=predictions,
            training_losses=training_losses,
            base_train_losses=self.base_train_losses,
            ae_train_losses=self.ae_train_losses,
        )
    

        