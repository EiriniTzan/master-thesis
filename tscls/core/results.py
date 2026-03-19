from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(slots=True)
class StepResult:
    """
    Output of a single step of the stream pipeline.
    """

    sample_index: int
    prediction: float
    latent_vector: np.ndarray
    reconstruction_error: float
    drift_detected: bool
    training_loss: Optional[float] = None


@dataclass(slots=True)
class PipelineResult:
    """
    Result of one full pipeline run.

    Attributes
    ----------
    step_results : list[StepResult]
        Per-sample results collected during stream processing.
    drift_points : list[int]
        Indices of samples at which drift was detected.
    reconstruction_errors : list[float]
        Reconstruction errors over the full stream.
    predictions : list[float]
        Predictions produced over the full stream.
    training_losses : list[float]
        Online training losses collected during stream processing.
    base_train_losses : list[float]
        Training losses of the reference DNN during the reference phase.
    ae_train_losses : list[float]
        Training losses of the autoencoder during the reference phase.
    """

    step_results: list[StepResult] = field(default_factory=list)
    drift_points: list[int] = field(default_factory=list)
    reconstruction_errors: list[float] = field(default_factory=list)
    predictions: list[float] = field(default_factory=list)
    training_losses: list[float] = field(default_factory=list)
    base_train_losses: list[float] = field(default_factory=list)
    ae_train_losses: list[float] = field(default_factory=list)