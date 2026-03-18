from dataclasses import dataclass, field


@dataclass(slots=True)
class DNNModelConfig:
    """
    Configuration of DNN models (1 & 2).

    Attributes
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        Sizes of the hidden layers.
    output_dim : int
        Number of output units.
    """

    input_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    output_dim: int = 1


@dataclass(slots=True)
class AutoencoderConfig:
    """
    Configuration of the autoencoder detector.

    Attributes
    ----------
    input_dim : int
        Dimensionality of the latent representation.
    encoder_dims : list[int]
        Hidden-layer dimensions of the encoder network.
    decoder_dims : list[int]
        Hidden-layer dimensions of the decoder network.
    threshold_k : float
        Multiplicative factor used by the thresholding rule.
    """

    input_dim: int
    encoder_dims: list[int]
    decoder_dims: list[int]
    threshold_k: float = 3.0


@dataclass(slots=True)
class StreamConfig:
    """
    Configuration of the adaptive stream model.

    Attributes
    ----------
    freeze_until_layer : int
        Hidden layers with index smaller than this value are frozen.
    """

    freeze_before_layer: int = 2


@dataclass(slots=True)
class OptimizationConfig:
    """
    Configuration of loss functions, optimizers, learning rates,
    and learning-rate adaptation hyperparameters.

    Attributes
    ----------
    base_loss_name : str
        Loss function used for the reference DNN.
    ae_loss_name : str
        Loss function used for the autoencoder.
    stream_loss_name : str
        Loss function used for the adaptive DNN.
    base_optimizer_name : str
        Optimizer used for the reference DNN.
    ae_optimizer_name : str
        Optimizer used for the autoencoder.
    stream_optimizer_name : str
        Optimizer used for the adaptive DNN.
    base_learning_rate : float
        Initial learning rate for the reference DNN.
    ae_learning_rate : float
        Learning rate for the autoencoder.
    stream_learning_rate : float
        Initial learning rate for the adaptive DNN.
    gamma1 : float
        Learning-rate decay factor for the reference DNN.
    s1 : int
        Step size controlling learning-rate updates during reference training.
    gamma2 : float
        Learning-rate decay factor for the adaptive DNN.
    s2 : int
        Step size controlling learning-rate updates during online adaptation.
    """

    base_loss_name: str = "bce_with_logits"
    ae_loss_name: str = "mse"
    stream_loss_name: str = "bce_with_logits"

    base_optimizer_name: str = "sgd"
    ae_optimizer_name: str = "adam"
    stream_optimizer_name: str = "adam"

    base_learning_rate: float = 1e-2
    ae_learning_rate: float = 1e-4
    stream_learning_rate: float = 1e-4

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
    base_epochs : int
        Number of epochs used to train the reference DNN.
    ae_epochs : int
        Number of epochs used to train the autoencoder.
    """

    base_epochs: int = 20
    ae_epochs: int = 10


@dataclass(slots=True)
class PipelineConfig:
    """
    Global configuration of the DNN + AE drift detection pipeline.

    Attributes
    ----------
    model : ModelConfig
        Configuration of the DNN models.
    autoencoder : AutoencoderConfig
        Configuration of the autoencoder detector.
    stream : StreamConfig
        Configuration of the stream adaptation phase.
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
    stream: StreamConfig = field(default_factory=StreamConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str | None = None