from dataclasses import dataclass, field


@dataclass(slots=True)
class DNNModelConfig:
    """
    Configuration of DNN models (1 & 2).

    Attributes
    ----------
    layer_sizes : list[int]
        Full list of layer sizes from input to output,
        e.g. ``[3, 256, 128, 64, 1]``. Must have at least three elements.
    activation : str
        Activation function applied after each hidden layer.
    """

    layer_sizes: list[int]
    activation: str = "relu"


@dataclass(slots=True)
class AutoencoderConfig:
    """
    Configuration of the autoencoder detector.

    Attributes
    ----------
    encoder_sizes : list[int]
        Layer sizes for the encoder, from input (latent dim) to bottleneck.
        Must have at least three elements, e.g. ``[64, 32, 8]``.
        The decoder is built as the mirror image automatically.
    activation : str
        Activation function used between layers (``"relu"``, ``"tanh"``,
        or ``"snake"``).
    threshold_k : float
        Multiplicative factor used by the k-sigma thresholding rule.
    """

    encoder_sizes: list[int]
    activation: str = "relu"
    threshold_k: float = 3.0


@dataclass(slots=True)
class OptimizationConfig:
    """
    Configuration of loss functions, optimizers, learning rates,
    and learning-rate adaptation hyperparameters.

    Attributes
    ----------
    model1_loss_name : str
        Loss function used for Model 1 (reference DNN).
    model3_loss_name : str
        Loss function used for Model 3 (autoencoder).
    model2_loss_name : str
        Loss function used for Model 2 (adaptive DNN).
    model1_optimizer_name : str
        Optimizer used for Model 1.
    model3_optimizer_name : str
        Optimizer used for Model 3.
    model2_optimizer_name : str
        Optimizer used for Model 2.
    model1_learning_rate : float
        Initial learning rate for Model 1.
    model3_learning_rate : float
        Learning rate for Model 3.
    model2_learning_rate : float
        Initial learning rate for Model 2.
    gamma1 : float
        Learning-rate decay factor for Model 1.
    s1 : int
        Step size controlling learning-rate updates during reference training.
    gamma2 : float
        Learning-rate decay factor for Model 2.
    s2 : int
        Step size controlling learning-rate updates during online adaptation.
    """

    model1_loss_name: str = "bce_with_logits"
    model3_loss_name: str = "mse"
    model2_loss_name: str = "bce_with_logits"

    model1_optimizer_name: str = "sgd"
    model3_optimizer_name: str = "adam"
    model2_optimizer_name: str = "adam"

    model1_learning_rate: float = 1e-2
    model3_learning_rate: float = 1e-4
    model2_learning_rate: float = 1e-4

    gamma1: float = 0.9
    s1: int = 20

    gamma2: float = 1.0
    s2: int = 10


@dataclass(slots=True)
class TrainingConfig:
    """
    Configuration of training duration.

    Attributes
    ----------
    model1_epochs : int
        Number of epochs used to train Model 1 (reference DNN).
    model3_epochs : int
        Number of epochs used to train Model 3 (autoencoder).
    batch_size : int
        Number of samples per mini-batch during offline training.
        Applies to both Model 1 and Model 3.
    """

    model1_epochs: int = 20
    model3_epochs: int = 10
    batch_size: int = 256


@dataclass(slots=True)
class PipelineConfig:
    """
    Global configuration of the DNN + AE drift detection pipeline.

    Attributes
    ----------
    model : DNNModelConfig
        Configuration of the DNN models.
    autoencoder : AutoencoderConfig
        Configuration of the autoencoder detector.
    optimization : OptimizationConfig
        Configuration of optimization and learning-rate adaptation.
    training : TrainingConfig
        Configuration of training duration.
    device : str | None
        Device on which the pipeline will run. If None, the pipeline
        selects the device automatically.
    """

    model: DNNModelConfig
    autoencoder: AutoencoderConfig
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str | None = None