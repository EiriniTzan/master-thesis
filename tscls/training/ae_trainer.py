import torch
import torch.nn as nn

from tscls.models.autoencoder import Autoencoder


class AETrainer:
    """
    Trainer for the autoencoder drift detector.
    This trainer fits the autoencoder on the latent representations
    extracted from the reference DNN and computes reconstruction
    errors used for drift detection.
    """

    def __init__(
        self,
        loss_function: nn.Module,
        device: str | None = None,
    ) -> None:
        """
        Initialize the AETrainer instance.

        Parameters
        ----------
        loss_function : nn.Module
            Loss function used for autoencoder training.
        device : str | None, optional
            Device on which computations should be performed. If None,
            it is selected automatically ("cuda" if available,
            otherwise "cpu").
        """

        self.loss_function = loss_function
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(
        self,
        autoencoder: Autoencoder,
        reference_latents: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        batch_size: int,
    ) -> list[float]:
        """
        Train the autoencoder on the reference latent representations.

        Parameters
        ----------
        autoencoder : Autoencoder
            The autoencoder model to be trained.
        reference_latents : torch.Tensor
            Reference latent representations from the DNN.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training.
        epochs : int
            The number of epochs to train the model.
        batch_size : int
            Number of samples per mini-batch.

        Returns
        -------
        list[float]
            A list of mean training losses, one value per epoch.
        """

        autoencoder.to(self.device)
        autoencoder.train()

        reference_latents = reference_latents.to(self.device)
        n = reference_latents.shape[0]
        losses: list[float] = []

        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            latents_shuffled = reference_latents[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                batch = latents_shuffled[start : start + batch_size]

                optimizer.zero_grad()
                reconstructed = autoencoder(batch)
                loss = self.loss_function(reconstructed, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            losses.append(epoch_loss / n_batches)

        return losses

    