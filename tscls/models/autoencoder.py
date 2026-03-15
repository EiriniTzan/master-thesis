from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """
    A fully connected autoencoder used for drift detection.
    The model consists of an encoder and a decoder neural network.
    It receives the latent representation produced by the classifier
    (output of the last hidden layer) and learns to reconstruct it.
    The reconstruction error between the input latent vector and its
    reconstruction is used as a signal for concept drift.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ) -> None:
        """
        Initialize the Autoencoder instance.

        Parameters
        ----------
        input_dim : int
            The number of input features.
        embedding_dim : int
            The number of latent features.

        Returns
        -------
        None
        """
        
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Autoencoder instance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The reconstructed input tensor.
        """

        z = self.encoder(x)
        reconstruction = self.decoder(z)

        return reconstruction

    def reconstruction_error(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the reconstruction error of the autoencoder for a given input tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the reconstructed input tensor and the reconstruction error.
        """

        reconstruction = self.forward(x)
        error = F.mse_loss(reconstruction, x, reduction = "none").mean(dim=1)

        return reconstruction, error