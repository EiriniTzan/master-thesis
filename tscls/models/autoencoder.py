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
            encoder_dims: list[int],
            decoder_dims: list[int],
    ) -> None:
        """
        Initialize the Autoencoder instance.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        encoder_dims : list[int]
            List of hidden layer dimensions for the encoder network.
        decoder_dims : list[int]
            List of hidden layer dimensions for the decoder network.
        """
        
        super().__init__()

        self.input_dim = input_dim
        self.encoder_dims = encoder_dims
        self.decoder_dims = decoder_dims

        self.encoder = self._build_encoder(
            input_dim = self.input_dim,
            encoder_dims = self.encoder_dims,
        )

        self.decoder = self._build_decoder(
            encoder_output_dim = self.encoder_dims[-1],
            decoder_dims = self.decoder_dims
        )

    def _build_encoder(
        self,
        input_dim: int,
        encoder_dims: list[int],
    ) -> nn.Sequential:    
        """
        Build the encoder network.

        Parameters
        ----------
        input_dim : int
            The number of input features.
        encoder_dims : list[int]
            List of hidden layer dimensions for the encoder network.

        Returns
        -------
        nn.Sequential
            The encoder network.
        """
        
        layers : list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in encoder_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim

        return nn.Sequential(*layers)

    def _build_decoder(
        self,
        encoder_output_dim: int,
        decoder_dims: list[int],
    ) -> nn.Sequential:
        """
        Build the decoder network.

        Parameters
        ----------
        encoder_output_dim : int
            The number of output features of the encoder network.
        decoder_dims : list[int]
            List of hidden layer dimensions for the decoder network.

        Returns
        -------
        nn.Sequential
            The decoder network.
        """

        layers : list[nn.Module] = []
        previous_dim = encoder_output_dim

        for hidden_dim in decoder_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            previous_dim = hidden_dim
        
        layers.append(nn.Linear(previous_dim, self.input_dim))

        return nn.Sequential(*layers)

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