"""Shared helpers for the SEA drift-detection notebooks."""

import numpy as np
import matplotlib.pyplot as plt


def configure_matplotlib() -> None:
    """Apply the standard rcParams used across all notebooks."""
    plt.rcParams.update({
        "font.size":        13,
        "axes.titlesize":   14,
        "axes.labelsize":   13,
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "legend.fontsize":  11,
    })


def rolling_mean(
    arr: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a causal rolling mean via convolution.

    Parameters
    ----------
    arr :
        1-D array of values (e.g. per-sample accuracy, agreement signal).
    window :
        Number of samples in the rolling window.

    Returns
    -------
    values :
        Rolling-mean values, length ``len(arr) - window + 1``.
    indices :
        Corresponding stream indices (causal: index of the last sample
        in each window).
    """
    values  = np.convolve(arr, np.ones(window) / window, mode="valid")
    indices = np.arange(window - 1, window - 1 + len(values))
    return values, indices


def add_drift_markers(
    ax,
    drift_pos: int,
    drift_points: list[int],
    *,
    first_label: str | None = None,
) -> None:
    """
    Annotate an axis with true-drift and detected-drift vertical lines.

    Parameters
    ----------
    ax :
        Matplotlib axis to annotate.
    drift_pos :
        Stream index of the true concept drift (green solid line).
    drift_points :
        Stream indices at which drift was detected (semi-transparent red
        lines). The first detection also gets an opaque dotted red line
        when ``first_label`` is provided.
    first_label :
        Legend label for the first-detection marker. Pass ``None`` to
        suppress the first-detection highlight.
    """
    ax.axvline(
        drift_pos,
        color="green",
        linestyle="-",
        linewidth=2.0,
        label=f"True drift at {drift_pos}",
    )
    for dp in drift_points:
        ax.axvline(dp, color="red", alpha=0.15, linewidth=0.8)
    if drift_points and first_label is not None:
        ax.axvline(
            drift_points[0],
            color="red",
            linewidth=2.0,
            linestyle=":",
            label=first_label,
        )


def plot_sea_data(
    axes,
    X_ref:     np.ndarray,
    y_ref:     np.ndarray,
    X_stream:  np.ndarray,
    y_stream:  np.ndarray,
    theta_old: float,
    theta_new: float,
    f_old:     int,
    f_new:     int,
    drift_pos: int,
) -> None:
    """
    Render the two-panel SEA concept scatter plot.

    Top panel shows reference data (single concept, one theta line).
    Bottom panel shows stream data with both theta lines and a true-drift
    marker. Both panels plot ``x_0 + x_1`` on the y-axis.

    Parameters
    ----------
    axes :
        Array of two Matplotlib axes (top, bottom).
    X_ref, y_ref :
        Reference feature matrix and binary labels.
    X_stream, y_stream :
        Stream feature matrix and binary labels.
    theta_old :
        Decision threshold before drift.
    theta_new :
        Decision threshold after drift.
    f_old, f_new :
        SEA concept function IDs (used in titles).
    drift_pos :
        Stream index of the true concept drift.
    """
    ax           = axes[0]
    idx_ref      = np.arange(len(y_ref))
    feat_sum_ref = X_ref[:, 0] + X_ref[:, 1]
    ax.scatter(
        idx_ref[y_ref == 1],
        feat_sum_ref[y_ref == 1],
        s=5, alpha=0.4, c="steelblue", label="Class 1",
    )
    ax.scatter(
        idx_ref[y_ref == 0],
        feat_sum_ref[y_ref == 0],
        s=5, alpha=0.4, c="salmon", label="Class 0",
    )
    ax.axhline(
        theta_old,
        color="black", linestyle="--", linewidth=1.5,
        label=f"θ = {theta_old}",
    )
    ax.set_title(f"Reference data — SEA concept f{f_old} (θ = {theta_old})")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("$x_0 + x_1$")
    ax.legend(markerscale=3, loc="upper right")

    ax              = axes[1]
    idx_stream      = np.arange(len(y_stream))
    feat_sum_stream = X_stream[:, 0] + X_stream[:, 1]
    ax.scatter(
        idx_stream[y_stream == 1],
        feat_sum_stream[y_stream == 1],
        s=5, alpha=0.4, c="steelblue", label="Class 1",
    )
    ax.scatter(
        idx_stream[y_stream == 0],
        feat_sum_stream[y_stream == 0],
        s=5, alpha=0.4, c="salmon", label="Class 0",
    )
    ax.axhline(
        theta_old,
        color="gray", linestyle="--", linewidth=1.5,
        label=f"θ before drift = {theta_old}",
    )
    ax.axhline(
        theta_new,
        color="black", linestyle="--", linewidth=1.5,
        label=f"θ after drift = {theta_new}",
    )
    ax.axvline(
        drift_pos,
        color="red", linestyle="-", linewidth=2.0,
        label=f"True drift at index {drift_pos}",
    )
    ax.set_title(
        f"Stream data — SEA f{f_old}→f{f_new}, drift at index {drift_pos}"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("$x_0 + x_1$")
    ax.legend(markerscale=3, loc="upper right")


def collect_capymoa_samples(
    stream,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Collect ``n`` instances from a capymoa stream.

    Parameters
    ----------
    stream :
        Any capymoa stream (``NumpyStream``, ``DriftStream``, ``SEA``, …)
        that implements ``has_more_instances`` and ``next_instance``.
    n :
        Maximum number of samples to collect. Collection stops early if
        the stream runs out.

    Returns
    -------
    X : np.ndarray of shape (n, num_features), dtype float32
    y : np.ndarray of shape (n,), dtype float32
        Integer class indices cast to float32.
    """
    X_list, y_list = [], []
    i = 0
    while stream.has_more_instances() and i < n:
        inst = stream.next_instance()
        X_list.append(inst.x)
        y_list.append(inst.y_index)
        i += 1
    return (
        np.array(X_list, dtype=np.float32),
        np.array(y_list, dtype=np.float32),
    )
