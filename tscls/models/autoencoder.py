from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tscls.models.builders import AutoencoderBuilder


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder used for drift detection.

    Receives the latent representation produced by the classifier
    (output of the last hidden layer) and learns to reconstruct it.
    The reconstruction error is used as a concept-drift signal.

    The encoder and decoder are built by ``AutoencoderBuilder``.
    """

    def __init__(
        self,
        encoder_sizes: list[int],
        activation: str = "relu",
        **activation_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        encoder_sizes : list[int]
            Layer sizes for the encoder, e.g. ``[64, 32, 8]``.
            Must have at least three elements. The decoder mirrors this.
        activation : str
            Activation between layers (``"relu"``, ``"tanh"``).
        **activation_kwargs :
            Extra arguments forwarded to the activation builder.
        """
        super().__init__()
        self.encoder, self.decoder = AutoencoderBuilder(
            encoder_sizes,
            activation,
            **activation_kwargs
        )()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor.

        Returns
        -------
        torch.Tensor
            Reconstructed latent tensor.
        """
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-sample reconstruction error (mean MSE over latent dimensions).

        Parameters
        ----------
        x : torch.Tensor
            Input latent tensor of shape ``(n, latent_dim)``.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(reconstruction, error)`` where ``error`` has shape ``(n,)``.
        """
        reconstruction = self.forward(x)
        error = F.mse_loss(reconstruction, x, reduction="none").mean(dim=1)
        return reconstruction, error
