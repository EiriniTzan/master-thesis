from dataclasses import dataclass

import torch

from tscls.detection.threshold import ThresholdRule
from tscls.models.autoencoder import Autoencoder


@dataclass(slots=True)
class DetectionResult:
    """
    Output of a single drift detection step.
    """

    reconstruction_error: float
    drift_detected: bool


class AEDetector:
    """
    Autoencoder-based drift detector.

    This detector uses a trained autoencoder to compute the reconstruction
    error of a latent representation and applies a sigma-based thresholding
    rule to decide whether the sample is drifted.
    """

    def __init__(
        self,
        autoencoder: Autoencoder,
        threshold_rule: ThresholdRule,
        device: str | None = None,
    ) -> None:
        """
        Initialize the drift detector.

        Parameters
        ----------
        autoencoder : Autoencoder
            The pre-trained autoencoder model used for reconstruction error calculation.
        threshold_rule : ThresholdRule
            The thresholding rule used to decide whether a sample is drifted or not.
        device : str | None, optional
            The device on which computations should be performed. If None,
            it will be determined automatically ("cuda" if available, otherwise "cpu").
        """

        self.autoencoder = autoencoder
        self.threshold_rule = threshold_rule
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def detect(self, latent: torch.Tensor) -> DetectionResult:
        """
        Perform a single drift detection step.

        Parameters
        ----------
        latent : torch.Tensor
            The latent representation of the input sample to be checked for drift.

        Returns
        -------
        DetectionResult
            A DetectionResult object containing the reconstruction error and a flag indicating
            whether drift was detected.
        """

        self.autoencoder.to(self.device)
        self.autoencoder.eval()

        latent = latent.to(self.device)

        with torch.no_grad():
            _, error = self.autoencoder.reconstruction_error(latent)

        error_val = float(error.item())
        drift_detected = self.threshold_rule.is_outside_bounds(error)

        return DetectionResult(
            reconstruction_error = error_val,
            drift_detected = drift_detected,
        )