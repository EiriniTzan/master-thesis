from abc import ABC, abstractmethod

import numpy as np


class SyntheticDataset(ABC):
    """
    Abstract base class for synthetic classification datasets.

    Subclasses must declare the fixed dimensionality of the feature
    space and the number of output classes, and implement generate().
    """

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of input features per sample."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of distinct class labels."""

    @abstractmethod
    def generate(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a labelled dataset.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.
        seed : int | None, optional
            Seed for the random number generator.

        Returns
        -------
        X : np.ndarray of shape (n_samples, num_features), dtype float32
            Feature matrix.
        y : np.ndarray of shape (n_samples,), dtype float32
            Label vector.
        """
