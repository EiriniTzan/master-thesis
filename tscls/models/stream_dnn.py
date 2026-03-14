from copy import deepcopy
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from tscls.models.base_dnn import DNNBase


class StreamDNN(nn.Module):
    """
    Online streaming DNN model used during the drift detection phase.
    This model starts from a pre-trained reference DNN and freezes the early hidden
    layers. However, it keeps the last hidden layer and output layer trainable.
    """

    def __init__(
        self,
        base_model: DNNBase,
        freeze_before_layer: int,
    ) -> None:
        super().__init__()

        self.stream_model = deepcopy(base_model)
        self.freeze_hidden_layers(freeze_before_layer)

    def freeze_hidden_layers(self, freeze_before_layer: int) -> None:
        """
        Freeze all hidden layers up to the given layer index.
        The output layer will always be trainable.
        Parameters
        ----------
        freeze_before_layer : int
            The index of the last hidden layer to be frozen.
        """

        for layer_index, hidden_layer in enumerate(self.stream_model.hidden_layers):
            is_trainable = layer_index >= freeze_before_layer

            for param in hidden_layer.parameters():
                param.requires_grad = is_trainable

        for param in self.stream_model.output_layer.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        """
        Returns an iterable of the trainable parameters of the model.
    
        Yields
        -------
        Iterable[nn.Parameter]
            An iterable of the trainable parameters of the model.
        """

        return (
            param
            for param in self.parameters()
            if param.requires_grad
        )

    def forward_with_latent(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the StreamDNN instance returning both
        classifier output and last hidden layer activation.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            A tuple containing the classifier output and the last hidden layer activation.
        """

        return self.stream_model.forward_with_latent(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the StreamDNN instance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The classifier output.
        """

        return self.stream_model.forward(x)