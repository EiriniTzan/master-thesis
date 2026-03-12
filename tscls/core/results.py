from dataclasses import dataclass
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


