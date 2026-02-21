"""
ae_drift_simul.py

Purpose
-------
A simulation of concept drift detection using:
1) learned representations (last hidden layer embeddings) from a DNN
2) autoencoder reconstruction error as a drift signal (thresholding with 3-σ)

This isn't the full streaming pipeline of the paper.
It's a first step to validate the core mechanism.

"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    n_train_old: int = 7000
    n_eval_old: int = 2000
    n_eval_new: int = 2000

    # Drift (mean drift on first drift_dims features)
    drift_dims: int = 5
    drift_shift: float = 3.0

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
    X_old_eval = np.random.randn(cfg.n_eval_old, n).astype(np.float32) # not used yet

    X_new_eval = np.random.randn(cfg.n_eval_new, n).astype(np.float32)
    X_new_eval[:, : cfg.drift_dims] += cfg.drift_shift # mean drift
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
        self.fc2 = nn.Linear(hidden1, hidden2) # Last hidden layer
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
        # We calculate MSE per sample
        e = ((recon - zb) ** 2).mean(dim=1)
        errs.append(e)
    return torch.cat(errs, dim=0)

def run_pipeline(cfg: SimConfig): 
    """
    Run the pipeline for the drift detection experiment.

    Parameters
    ----------
    cfg : SimConfig
        The simulation configuration.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, float, float, float, float, torch.Tensor]
        A tuple containing the baseline (train) reconstruction errors, the new concept reconstruction errors,
        the threshold used for drift detection, the mean and standard deviation of the baseline reconstruction errors,
        the fraction of new samples flagged as drifted, and the per sample flags (drift/anomaly flags).
    """
    set_seed(cfg.seed)
    device = get_device()

    # Data (old vs new)
    X_old_train_np, X_old_eval_np, X_new_eval_np = gener_synthetic(cfg)
    y_old_train_np = gener_regression_targets(X_old_train_np, cfg.target_noise_std)

    X_old_train = to_tensor(X_old_train_np, device)
    y_old_train = to_tensor(y_old_train_np, device)
    X_old_eval = to_tensor(X_old_eval_np, device)
    X_new_eval = to_tensor(X_new_eval_np, device)

    # Pretrain MLP on old concept
    mlp = MLPRepresentation(in_dim=cfg.n_features).to(device)
    train_mlp(mlp, X_old_train, y_old_train, cfg)

    # Extract embeddings
    Z_train_old = extract_embeddings(mlp, X_old_train) # baseline embeddings (train)
    Z_new = extract_embeddings(mlp, X_new_eval)

    # Train AE on baseline embeddings
    ae = AutoEncoder(in_dim=Z_train_old.shape[1], latent=16).to(device)
    train_ae(ae, Z_train_old, cfg)

    # Reconstruction errors
    err_train = reconstr_errors(ae, Z_train_old) 
    err_new = reconstr_errors(ae, Z_new) 

    # Threshold
    mu = err_train.mean().item()
    sigma = err_train.std(unbiased=False).item()
    threshold = mu + cfg.sigma_k * sigma

    # Per sample flags (drift/anomaly flags)
    flagged = (err_new > threshold)
    flag_fraction = flagged.float().mean().item()


    return err_train, err_new, threshold, mu, sigma, flag_fraction, flagged

def sensitivity_curve(cfg: SimConfig, drift_values: list[float]) -> list[float]:    
    """
    Calculate the sensitivity curve of the drift detection algorithm.

    The sensitivity curve is a plot of the fraction of new samples flagged as drifted
    vs the drift shift value. This curve can be used to evaluate the performance
    of the algorithm and to select the optimal hyperparameters.

    Parameters
    ----------
    cfg : SimConfig
        The simulation configuration.
    drift_values : list[float]
        The list of drift shift values for which the sensitivity curve should be calculated.

    Returns
    -------
    list[float]
        A list containing the fraction of new samples flagged as drifted for each drift shift value.
    """
    fracs_above_thr = []
    for d in drift_values:
        cfg_d = SimConfig(
            seed=cfg.seed,
            n_features=cfg.n_features,
            n_train_old=cfg.n_train_old,
            n_eval_old=cfg.n_eval_old,
            n_eval_new=cfg.n_eval_new,
            drift_dims=cfg.drift_dims,
            drift_shift=d,
            target_noise_std=cfg.target_noise_std,
            mlp_epochs=cfg.mlp_epochs,
            ae_epochs=cfg.ae_epochs,
            batch_size=cfg.batch_size,
            lr=cfg.lr,
            sigma_k=cfg.sigma_k,
        )
        _, _, _, _, _, flag_fraction, _ = run_pipeline(cfg_d)
        fracs_above_thr.append(flag_fraction)
    return fracs_above_thr

def plot_error_hist(err_old: torch.Tensor, err_new: torch.Tensor, threshold: float, title: str) -> None:
    """
    Plot a histogram of the old and new reconstruction errors, along with a vertical line
    indicating the threshold used for drift detection.

    Parameters
    ----------
    err_old : torch.Tensor
        The reconstruction errors of the old concept.
    err_new : torch.Tensor
        The reconstruction errors of the new concept.
    threshold : float
        The threshold used for drift detection.
    title : str
        The title of the plot.
    """
    eold = err_old.detach().cpu().numpy()
    enew = err_new.detach().cpu().numpy()
    plt.figure()
    plt.hist(eold, bins=40, alpha=0.5, label="Old")
    plt.hist(enew, bins=40, alpha=0.5, label="New")
    plt.axvline(threshold, linestyle="--", label="Threshold")
    plt.title(title)
    plt.xlabel("Reconstruction error")
    plt.ylabel("Count")
    plt.legend()
    plt.show()

def main() -> None:
    """
    Main function for running the experiment.
    It runs the pipeline with the default configuration, prints the results,
    and plots the reconstruction error histograms and sensitivity curve.
    """
    cfg = SimConfig()
    device = get_device()
    print(f"Device: {device}")

    err_base, err_new, threshold, mu, sigma, flag_fraction, flagged = run_pipeline(cfg)

    print("\n--- Drift Detection results ---")
    print(f"Baseline (train) error mean = {mu:.6f}")
    print(f"Baseline (train) error std  = {sigma:.6f}")
    print(f"{cfg.sigma_k:.1f}-sigma threshold (upper) = {threshold:.6f}")
    print(f"Flagged fraction in new = {flag_fraction:.3f}")

    # Plot error histograms for the baseline cfg
    plot_error_hist(err_base, err_new, threshold, title=f"Reconstruction error (drift_shift={cfg.drift_shift})")

    # Plot sensitivity curve
    drift_values = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    fracs = sensitivity_curve(cfg, drift_values)

    plt.figure()
    plt.plot(drift_values, fracs, marker="o")
    plt.xlabel("drift_shift")
    plt.ylabel("fraction(new error > threshold)")
    plt.title("Sensitivity: drift_shift vs detection rate")
    plt.ylim(-0.05, 1.05)
    plt.show()


if __name__ == "__main__":
    main()

