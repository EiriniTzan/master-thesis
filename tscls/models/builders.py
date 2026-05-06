from typing import Any, Tuple

import torch
import torch.nn as nn


class ActivationBuilder:
    """
    A builder for activation functions.

    Parameters
    ----------
    name : str
        Name of the activation function (``"relu"``, ``"tanh"``).
    **kwargs :
        Additional arguments forwarded to the activation
        (e.g. ``alpha_init`` for Snake).
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        self.name = name.lower()
        self.kwargs = kwargs

    def build(self, in_features: int) -> nn.Module:
        """
        Parameters
        ----------
        in_features : int
            Number of input features (needed for feature-wise activations).

        Returns
        -------
        nn.Module
        """
        if self.name == "tanh":
            return nn.Tanh()
        if self.name == "relu":
            return nn.ReLU()
        raise ValueError(f"Unknown activation function: {self.name!r}")


def initialize_linear_layers(module: nn.Module) -> None:
    """Apply Xavier-uniform init to all Linear layers in a module, zero bias."""
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class FeedforwardBuilder:
    """
    Builder for a feedforward network.

    All layers except the last are followed by the configured activation.
    The last layer is a linear projection with no activation.
    All linear layers are Xavier-uniform initialised.

    Parameters
    ----------
    layer_sizes : list[int]
        Layer sizes from input to output, e.g. ``[64, 32, 8]``.
        Must have at least three elements.
    activation : str
        Activation function between layers. Default ``"relu"``.
    **activation_kwargs :
        Extra arguments forwarded to the activation builder.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        **activation_kwargs: Any,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 elements.")
        self.layer_sizes = layer_sizes
        self._activation_builder = ActivationBuilder(activation, **activation_kwargs)

    def __call__(self) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(len(self.layer_sizes) - 2):
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            layers.append(self._activation_builder.build(self.layer_sizes[i + 1]))
        layers.append(nn.Linear(self.layer_sizes[-2], self.layer_sizes[-1]))

        net = nn.Sequential(*layers)
        initialize_linear_layers(net)
        return net


class AutoencoderBuilder:
    """
    Builder for a symmetric autoencoder.

    ``layer_sizes`` describes the encoder (input → hidden … → bottleneck);
    the decoder is the mirror image. Calling the builder returns
    ``(encoder, decoder)`` as a pair of ``nn.Sequential`` networks.

    Parameters
    ----------
    layer_sizes : list[int]
        Encoder layer sizes, e.g. ``[64, 32, 8]``.
        Must have at least three elements.
    activation : str
        Activation function between layers. Default ``"relu"``.
    **activation_kwargs :
        Extra arguments forwarded to the activation builder.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        **activation_kwargs: Any,
    ) -> None:
        self._encoder_builder = FeedforwardBuilder(
            layer_sizes,
            activation,
            **activation_kwargs
        )
        self._decoder_builder = FeedforwardBuilder(
            list(reversed(layer_sizes)),
            activation,
            **activation_kwargs
        )

    def __call__(self) -> Tuple[nn.Sequential, nn.Sequential]:
        """
        Returns
        -------
        tuple[nn.Sequential, nn.Sequential]
            ``(encoder, decoder)``
        """
        return self._encoder_builder(), self._decoder_builder()
