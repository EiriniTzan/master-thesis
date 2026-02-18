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
    """
    Return a torch device (either "cuda" or "cpu") depending on whether a CUDA device is available.

    Returns
    -------
    torch.device
        The device to be used for computations.
    """
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

    # Drift (mean drift on first drift_dims features)
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
    Generate synthetic data for concept drift detection experiment.

    Parameters
    ----------
    cfg : SimConfig
        Simulation configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple containing old training data, old evaluation data, and new evaluation data
    """
    n = cfg.n_features
    X_old_train = np.random.randn(cfg.n_train_old, n).astype(np.float32)
    X_old_eval = np.random.randn(cfg.n_eval_old, n).astype(np.float32)

    X_new_eval = np.random.randn(cfg.n_eval_new, n).astype(np.float32)
    X_new_eval[:, : cfg.drift_dims] += cfg.drift_shift  # mean drift
    return X_old_train, X_old_eval, X_new_eval


def gener_regression_targets(X: np.ndarray, noise_std: float) -> np.ndarray:
    
    """
    Generate regression targets for a given dataset.

    Parameters
    ----------
    X : np.ndarray
        The input dataset.
    noise_std : float
        The standard deviation of the noise to be added to the targets.

    Returns
    -------
    np.ndarray
        The generated regression targets.
    """
    n = X.shape[1]
    w = np.linspace(0.5, 1.5, n).astype(np.float32)
    y = X @ w + noise_std * np.random.randn(X.shape[0]).astype(np.float32)
    return y.reshape(-1, 1)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert a numpy array to a torch tensor and move it to the specified device.

    Parameters
    ----------
    x : np.ndarray
        The input numpy array.
    device : torch.device
        The device to which the tensor should be moved.

    Returns
    -------
    torch.Tensor
        The tensor created from the input numpy array.
    """
    return torch.from_numpy(x).to(device)


def batch_iter(X: torch.Tensor, y: torch.Tensor, batch_size: int):

    """
    Iterate over the dataset in batches.

    Parameters
    ----------
    X : torch.Tensor
        The input data
    y : torch.Tensor
        The target data
    batch_size : int
        The size of each batch

    Yields
    ------
    X_batch : torch.Tensor
        A batch of input data
    y_batch : torch.Tensor
        A batch of target data
    """
    n = X.shape[0]
    idx = torch.randperm(n, device=X.device)
    for i in range(0, n, batch_size):
        j = idx[i : i + batch_size]
        yield X[j], y[j]

# Models (MLP, Autoencoder)
class MLPRepresentation(nn.Module):
    """
    MLP that returns: yhat (regression head output),
    emb (last hidden layer activation)
    """
    def __init__(self, in_dim: int, hidden1: int = 64, hidden2: int = 32, out_dim: int = 1):
        """
        Initialize the MLP_representation instance.

        Parameters
        ----------
        in_dim : int
            The number of input features.
        hidden1 : int, optional
            The number of neurons in the first hidden layer. Defaults to 64.
        hidden2 : int, optional
            The number of neurons in the second hidden layer. Defaults to 32.
        out_dim : int, optional
            The number of output features. Defaults to 1.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)  # last hidden layer
        self.head = nn.Linear(hidden2, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the MLP_representation instance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the regression head output and the last hidden layer activation.
        """
        x = F.relu(self.fc1(x))
        emb = F.relu(self.fc2(x))
        yhat = self.head(emb)
        return yhat, emb
    

class AutoEncoder(nn.Module):
    def __init__(self, in_dim: int, latent: int = 16, hidden: int = 32):
        """
        Initialize the AutoEncoder instance.

        Parameters
        ----------
        in_dim : int
            The number of input features.
        latent : int, optional
            The number of latent features. Defaults to 16.
        hidden : int, optional
            The number of hidden features. Defaults to 32.
        """
        super().__init__()
        self.enc1 = nn.Linear(in_dim, hidden)
        self.enc2 = nn.Linear(hidden, latent)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, in_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AutoEncoder instance.

        Parameters
        ----------
        z : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The reconstructed input tensor.
        """
        h = F.relu(self.enc1(z))
        h = F.relu(self.enc2(h))
        h = F.relu(self.dec1(h))
        return self.dec2(h)
        
# Training
def train_mlp(model: MLPRepresentation, X: torch.Tensor, y: torch.Tensor, cfg: SimConfig) -> None:
    """
    Train an MLP_representation instance.

    Parameters
    ----------
    model : MLPRepresentation
        The model to be trained.
    X : torch.Tensor
        The input data.
    y : torch.Tensor
        The target data.
    cfg : SimConfig
        The simulation configuration.

    Returns
    -------
    None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    for ep in range(1, cfg.mlp_epochs + 1):
        losses = []
        for xb, yb in batch_iter(X, y, cfg.batch_size):
            optimizer.zero_grad(set_to_none=True)
            yhat, _ = model(xb)
            loss = F.mse_loss(yhat, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[MLP pretrain] epoch {ep:02d} | loss={float(np.mean(losses)):.4f}")


@torch.no_grad()
def extract_embeddings(model: MLPRepresentation, X: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """
    Extract last hidden layer embeddings from a given MLP_representation model.

    Parameters
    ----------
    model : MLPRepresentation
        The model from which to extract the embeddings.
    X : torch.Tensor
        The input data.
    batch_size : int, optional
        The batch size to be used for inference. Defaults to 512.

    Returns
    -------
    torch.Tensor
        The extracted embeddings. (size: [n_samples, embedding_dim])
    """
    model.eval()
    embs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i : i + batch_size]
        _, emb = model(xb)
        embs.append(emb)
    return torch.cat(embs, dim=0)

def train_ae(ae: AutoEncoder, Z: torch.Tensor, cfg: SimConfig) -> None:
    """
    Train an AutoEncoder instance.

    Parameters
    ----------
    ae : AutoEncoder
        The model to be trained.
    Z : torch.Tensor
        The input data.
    cfg : SimConfig
        The simulation configuration.

    Returns
    -------
    None
    """
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr)
    ae.train()
    for ep in range(1, cfg.ae_epochs + 1):
        losses = []
        idx = torch.randperm(Z.shape[0], device=Z.device)
        for i in range(0, Z.shape[0], cfg.batch_size):
            j = idx[i : i + cfg.batch_size]
            zb = Z[j]
            opt.zero_grad(set_to_none=True)
            recon = ae(zb)
            loss = F.mse_loss(recon, zb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        print(f"[AE train]     epoch {ep:02d} | loss={float(np.mean(losses)):.4f}")


@torch.no_grad()
def reconstr_errors(ae: AutoEncoder, Z: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    """
    Calculate the reconstruction errors of the AutoEncoder instance.

    Parameters
    ----------
    ae : AutoEncoder
        The AutoEncoder instance for which the reconstruction errors should be calculated.
    Z : torch.Tensor
        The input data.
    batch_size : int, optional
        The batch size to use for the reconstruction error calculation. Defaults to 512.

    Returns
    -------
    torch.Tensor
        The reconstruction errors of the AutoEncoder instance.
    """
    ae.eval()
    errs = []
    for i in range(0, Z.shape[0], batch_size):
        zb = Z[i : i + batch_size]
        recon = ae(zb)
        # we calculate MSE per sample
        e = ((recon - zb) ** 2).mean(dim=1)
        errs.append(e)
    return torch.cat(errs, dim=0)


