from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PipelineState:
    """
    Runtime state of the drift detection pipeline.
    """

    current_step: int = 0

    predictions: List[float] = field(default_factory=list)
    train_losses: List[float] = field(default_factory=list)

    reconstruction_errors: List[float] = field(default_factory=list)
    drift_points: List[int] = field(default_factory=list)

    threshold_mean: Optional[float] = None
    threshold_std: Optional[float] = None
    threshold_lower: Optional[float] = None
    threshold_upper: Optional[float] = None

    