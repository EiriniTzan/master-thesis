from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ThresholdStat:
    """
    Statistics estimated from the reference reconstruction errors.

    Attributes
    ----------
    mean : float
        Mean reconstruction error on the reference data.
    std : float
        Standard deviation of the reference reconstruction errors.
    lower : float
        Lower threshold bound.
    upper : float
        Upper threshold bound.
    """

    mean: float
    std: float
    lower: float
    upper: float


class ThresholdRule:
    """
    Sigma-based thresholding rule for reconstruction errors.
    This class estimates lower and upper bounds from reference
    reconstruction errors using the k-sigma rule:

        lower = mean - k * std
        upper = mean + k * std

    A new data point is classified as drift data point if its reconstruction error
    falls outside these bounds.
    """

    def __init__(self, k: float = 3.0) -> None:
        
        """
        Initialize a SigmaThreshold instance.

        Parameters
        ----------
        k : float, optional
            The number of standard deviations to use for thresholding.
            Defaults to 3.0.

        Raises
        -------
        ValueError
            If k is not a positive number.
        """
        if k <= 0:
            raise ValueError("k must be a positive number.")

        self.k = k
        self.stats: ThresholdStat | None = None

    def calibrate(self, reference_errors: torch.Tensor) -> ThresholdStat:
        """
        Calibrate the thresholding rule from reference reconstruction errors.

        Parameters
        ----------
        reference_errors : torch.Tensor
            Reference reconstruction errors from the autoencoder.

        Returns
        -------
        ThresholdStat
            The calibrated thresholding statistics.
        """

        reference_errors = reference_errors.float().view(-1)

        mean = float(reference_errors.mean().item())
        std = float(reference_errors.std(unbiased=False).item())
        lower = mean - self.k * std
        upper = mean + self.k * std

        self.stats = ThresholdStat(
            mean = mean,
            std = std,
            lower = lower,
            upper = upper,
        )

        return self.stats

    def is_outside_bounds(self, error: torch.Tensor) -> bool:
        """
        Check if a reconstruction error is outside the calibrated threshold bounds.

        Parameters
        ----------
        error : torch.Tensor
            The reconstruction error to check.

        Returns
        -------
        bool
            True if the error is outside the bounds, False otherwise.
        """
        
        if self.stats is None:
            raise RuntimeError("Threshold statistics have not been calibrated yet.")

        error_value = float(error.item())

        return error_value < self.stats.lower or error_value > self.stats.upper
