import torch

from tscls.core.results import PipelineResult
from tscls.core.sample import Sample
from tscls.pipeline.detector import Detector
from tscls.simulation.stream_simulation import StreamSimulator


class StreamMonitor:
    """
    Drives the online drift-detection loop.

    Accepts a trained Detector and a StreamSimulator, iterates over the
    stream sample by sample, and returns a PipelineResult aggregating all
    per-sample outcomes.
    """

    def __init__(self, detector: Detector) -> None:
        """
        Parameters
        ----------
        detector : Detector
            A Detector instance on which fit() has already been called.
        """

        self.detector = detector

    def run(self, stream: StreamSimulator) -> PipelineResult:
        """
        Iterate over the stream and collect drift-detection results.

        Parameters
        ----------
        stream : StreamSimulator
            The data stream to monitor.

        Returns
        -------
        PipelineResult
            Aggregated results: per-step details, drift points,
            reconstruction errors, predictions, and training losses.
        """

        stream.reset()

        drift_points: list[int] = []
        reconstruction_errors: list[float] = []
        predictions: list[float] = []
        training_losses: list[float] = []
        step_results = []

        while stream.has_next_sample():
            x, y, i = stream.next_sample()

            sample = Sample(
                x=(
                    x.detach().cpu().numpy()
                    if isinstance(x, torch.Tensor)
                    else x
                ),
                y=int(y.item()) if isinstance(y, torch.Tensor) else int(y),
                index=i,
            )

            step_result = self.detector.detect(sample)

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
            model1_train_losses=self.detector.model1_train_losses,
            model3_train_losses=self.detector.model3_train_losses,
        )
