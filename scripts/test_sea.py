#! /usr/bin/env python3

import numpy as np

from tscls.datasets.sea import SEA
from tscls.pipeline.detector import Detector
from tscls.pipeline.monitor import StreamMonitor
from tscls.pipeline.pipeline_config import (
    PipelineConfig,
    DNNModelConfig,
    AutoencoderConfig,
    OptimizationConfig,
    TrainingConfig,
)
from tscls.simulation.stream_simulation import StreamSimulator


def prepare_sea_data(
    n1: int = 5_000,
    n2: int = 5_000,
    drift_pos: int = 2_500,
    f1: int = 3,
    f2: int = 2,
    seed: int = 0,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Generate SEA data for a single-drift experiment.

    Parameters
    ----------
    n1 : int, optional
        Number of reference samples (concept f1, no drift).
        Defaults to 5000.
    n2 : int, optional
        Number of stream samples. Defaults to 5000.
    drift_pos : int, optional
        Index within the stream at which the concept changes from
        f1 to f2. Defaults to 2500.
    f1 : int, optional
        SEA concept index before the drift. Defaults to 3.
    f2 : int, optional
        SEA concept index after the drift. Defaults to 2.
    seed : int, optional
        Base seed for reproducibility. Defaults to 0.

    Returns
    -------
    (X1, y1) : tuple of np.ndarray
        Reference data (n1 samples, concept f1).
    (X2, y2) : tuple of np.ndarray
        Stream data (n2 samples, drift at drift_pos).
    """

    XY1 = SEA(f1, balanced=True).generate(n1, seed=seed)

    x_pre, y_pre = SEA(f1).generate(drift_pos, seed=seed + 1)
    x_post, y_post = SEA(f2).generate(n2 - drift_pos, seed=seed + 2)
    XY2 = (
        np.vstack([x_pre, x_post]),
        np.concatenate([y_pre, y_post]),
    )

    return XY1, XY2


def build_config(num_features: int) -> PipelineConfig:
    """
    Build a PipelineConfig for the DNN + AE drift detection pipeline.

    Parameters
    ----------
    num_features : int
        Number of input features.

    Returns
    -------
    PipelineConfig
    """

    return PipelineConfig(
        model=DNNModelConfig(
            layer_sizes=[num_features, 256, 128, 64, 1],
        ),
        autoencoder=AutoencoderConfig(
            encoder_sizes=[64, 32, 8],
            activation="relu",
            threshold_k=3.0,
        ),
        optimization=OptimizationConfig(
            base_loss_name="bce_with_logits",
            ae_loss_name="mse",
            stream_loss_name="bce_with_logits",
            base_optimizer_name="sgd",
            ae_optimizer_name="adam",
            stream_optimizer_name="sgd",
            base_learning_rate=1e-2,
            ae_learning_rate=1e-4,
            stream_learning_rate=1e-4,
            gamma1=0.9,
            s1=20,
            gamma2=1.0,
            s2=10,
        ),
        training=TrainingConfig(
            base_epochs=30,
            ae_epochs=10,
        ),
        device="cpu",
    )


def main() -> None:
    """
    Run the synthetic single-drift test using SEA concept data.
    """

    n1 = 5_000
    n2 = 5_000
    drift_pos = 2_500
    f_old = 3
    f_new = 2

    (X1, y1), (X2, y2) = prepare_sea_data(
        n1=n1,
        n2=n2,
        drift_pos=drift_pos,
        f1=f_old,
        f2=f_new,
        seed=0,
    )

    num_features = SEA(f_old).num_features

    config = build_config(num_features=num_features)
    detector = Detector(config=config)
    detector.fit(X1, y1)

    monitor = StreamMonitor(detector=detector)
    stream = StreamSimulator(data=X2, labels=y2)
    result = monitor.run(stream)

    print("Synthetic single-drift test completed (SEA concepts)")
    print(f"Number of input features: {num_features}")
    print(f"SEA concept before drift: f{f_old}  (theta={SEA(f_old).threshold})")
    print(f"SEA concept after drift:  f{f_new}  (theta={SEA(f_new).threshold})")
    print(f"Reference samples: {len(X1)}")
    print(f"Stream samples: {len(X2)}")
    print(f"True drift starts at stream index: {drift_pos}")
    print(f"Detected drift points (first 20): {result.drift_points[:20]}")
    print(f"Number of detected drift points: {len(result.drift_points)}")
    print(
        f"First 10 predictions: "
        f"{[round(p, 4) for p in result.predictions[:10]]}"
    )
    print(
        f"First 10 reconstruction errors: "
        f"{[round(e, 6) for e in result.reconstruction_errors[:10]]}"
    )

    if result.drift_points:
        first_detected = result.drift_points[0]
        print(f"First detected drift point: {first_detected}")
        print(f"Detection delay: {first_detected - drift_pos}")
    else:
        print("No drift point detected.")

    print(f"Last 3 Model 1 training losses: {result.base_train_losses[-3:]}")
    print(
        f"Last 3 autoencoder training losses: {result.ae_train_losses[-3:]}"
    )


if __name__ == "__main__":
    main()
