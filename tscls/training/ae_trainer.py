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

        Returns
        -------
        list[float]
            A list of training losses at each epoch.
        """

        autoencoder.to(self.device)
        autoencoder.train()

        reference_latents = reference_latents.to(self.device)
        losses: list[float] = []

        for _ in range(epochs):
            optimizer.zero_grad()

            reconstructed = autoencoder(reference_latents)
            loss = self.loss_function(reconstructed, reference_latents)

            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        return losses

    