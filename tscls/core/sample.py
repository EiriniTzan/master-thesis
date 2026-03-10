from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(slots=True)
class Sample:
    """
    Representation of a single sample in the simulated data stream.

    Attributes
    ----------
    x : np.ndarray
        Feature vector of the sample.
    y : Optional[int]
        Target label of the sample.
    index : Optional[int]
        Position of the sample in the stream.
    """

    x: np.ndarray
    y: Optional[int] = None
    index: Optional[int] = None