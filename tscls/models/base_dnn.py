from typing import Any, Tuple

import torch
import torch.nn as nn

from tscls.models.builders import FeedforwardBuilder


class DNNBase(nn.Module):
    """
    A fully connected feedforward deep neural network.
    Used as the base classifier (Model 1) in the drift detection pipeline.

    Built from a single ``layer_sizes`` list via ``FeedforwardBuilder``.
    Attributes
    ----------
    body : nn.Sequential
        All hidden layers except the last (with activations).
    last_hidden_layer : nn.Sequential
        The last hidden linear + its activation. Its output is the
        latent representation passed to the autoencoder.
    output_layer : nn.Linear
        Final linear projection with no activation.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        **activation_kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        layer_sizes : list[int]
            Full list of layer sizes, e.g. ``[3, 256, 128, 64, 1]``.
            Must have at least three elements.
        activation : str
            Activation function applied after each hidden layer.
        **activation_kwargs :
            Extra arguments forwarded to the activation builder.
        """
        super().__init__()

        net = FeedforwardBuilder(layer_sizes, activation, **activation_kwargs)()
        children = list(net.children())
        self.body = nn.Sequential(*children[:-3])
        self.last_hidden_layer = nn.Sequential(*children[-3:-1])
        self.output_layer: nn.Linear = children[-1]

    def forward_with_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(logits, latent)`` where ``latent`` is the output of
            ``last_hidden_layer``.
        """
        x = self.body(x)
        latent = self.last_hidden_layer(x)
        logits = self.output_layer(latent)
        return logits, latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward_with_latent(x)
        return logits
