from tscls.core.results import PipelineResult
from tscls.core.sample import Sample
from tscls.pipeline.detector import AEDriftDetector


class StreamMonitor:
    """
    Drives the online drift-detection loop.

    Accepts a trained AEDriftDetector and a capymoa-compatible stream,
    iterates over the stream sample by sample, and returns a PipelineResult
    aggregating all per-sample outcomes.
    """

    def __init__(self, detector: AEDriftDetector) -> None:
        """
        Parameters
        ----------
        detector : AEDriftDetector
            An AEDriftDetector instance on which fit() has already been called.
        """

        self.detector = detector

    def run(self, stream) -> PipelineResult:
        """
        Iterate over a capymoa stream and collect drift-detection results.

        Parameters
        ----------
        stream : capymoa Stream
            Any stream that implements the capymoa Stream interface
            (``restart``, ``has_more_instances``, ``next_instance``).
            E.g. ``NumpyStream``, ``DriftStream``, ``SEA``.

        Returns
        -------
        PipelineResult
            Aggregated results: per-step details, drift points,
            reconstruction errors, predictions, and training losses.
        """

        stream.restart()

        drift_points: list[int] = []
        reconstruction_errors: list[float] = []
        predictions: list[float] = []
        training_losses: list[float] = []
        step_results = []
        i = 0

        while stream.has_more_instances():
            instance = stream.next_instance()

            sample = Sample(
                x=instance.x,
                y=instance.y_index,
                index=i,
            )

            step_result = self.detector.detect(sample)

            step_results.append(step_result)
            reconstruction_errors.append(step_result.reconstruction_error)
            predictions.append(step_result.prediction)
            training_losses.append(step_result.training_loss)

            if step_result.drift_detected:
                drift_points.append(i)

            i += 1

        return PipelineResult(
            step_results=step_results,
            drift_points=drift_points,
            reconstruction_errors=reconstruction_errors,
            predictions=predictions,
            training_losses=training_losses,
            model1_train_losses=self.detector.model1_train_losses,
            model3_train_losses=self.detector.model3_train_losses,
        )
