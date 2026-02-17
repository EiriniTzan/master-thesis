"""
ae_drift_simul.py

Purpose
-------
A simulation of concept drift detection using:
1) learned representations (last hidden layer embeddings) from a small DNN
2) autoencoder reconstruction error as a drift signal (thresholding with 3-σ)

This is not the full streaming pipeline of the paper.
It's a first step to validate the core mechanism end-to-end.

"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurations
@dataclass(frozen=True)
class SimConfig:
    seed: int = 42

    # Data
    n_features: int = 20
    n_train_old: int = 8000
    n_eval_old: int = 2000
    n_eval_new: int = 2000

    # Drift (mean shift on first drift_dims features)
    drift_dims: int = 5
    drift_shift: float = 1.0

    # MLP pretrain task (simple regression target)
    target_noise_std: float = 0.1

    # Training
    mlp_epochs: int = 8
    ae_epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3

    # Threshold
    sigma_k: float = 3.0


def gener_synthetic(cfg: SimConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X_old_train, X_old_eval, X_new_eval
    """
    n = cfg.n_features
    X_old_train = np.random.randn(cfg.n_train_old, n).astype(np.float32)
    X_old_eval = np.random.randn(cfg.n_eval_old, n).astype(np.float32)

    X_new_eval = np.random.randn(cfg.n_eval_new, n).astype(np.float32)
    X_new_eval[:, : cfg.drift_dims] += cfg.drift_shift  # drift by mean shift
    return X_old_train, X_old_eval, X_new_eval


def gener_regression_targets(X: np.ndarray, noise_std: float) -> np.ndarray:
    """
    Simple supervised signal to force the MLP to learn non-trivial embeddings.
    y = Xw + noise
    """
    n = X.shape[1]
    w = np.linspace(0.5, 1.5, n).astype(np.float32)
    y = X @ w + noise_std * np.random.randn(X.shape[0]).astype(np.float32)
    return y.reshape(-1, 1)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x).to(device)


def batch_iter(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device)
    for i in range(0, n, batch_size):
        j = idx[i : i + batch_size]
        yield X[j], y[j]
