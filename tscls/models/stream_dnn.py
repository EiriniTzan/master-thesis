from copy import deepcopy
from typing import Tuple

import torch
import torch.nn as nn

from tscls.models.dnn_classifier import DNNClassifier


class StreamDNN(nn.Module):
    """
    Online streaming DNN used during the drift detection phase (Algorithm 3).

    Deep-copies the pre-trained Model 1, then freezes all layers except
    the last hidden linear (W(L)_F, b(L)_F). Only that layer is updated
    per sample so the AE can detect the resulting latent shift after drift.
    """

    def __init__(self, base_model: DNNClassifier) -> None:
        """
        Parameters
        ----------
        base_model : DNNClassifier
            The pre-trained reference DNN model.
        """

        super().__init__()

        self.stream_model = deepcopy(base_model)
        self._freeze_layers()

    def _freeze_layers(self) -> None:
        """
        Freeze all layers except the last hidden linear (Algorithm 3, paper).

        The first hidden layers and the output layer are frozen.
        Only W(L)_F and b(L)_F (last hidden linear) remain trainable
        so that the AE can detect the latent shift after drift.
        """

        for param in self.stream_model.parameters():
            param.requires_grad = False

        for param in self.stream_model.last_hidden_layer.parameters():
            param.requires_grad = True

        self._trainable_params: list[nn.Parameter] = list(
            self.stream_model.last_hidden_layer.parameters()
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Pre-computed trainable parameters (last hidden layer only)."""
        return self._trainable_params

    def forward_with_latent(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of StreamDNN returning both
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
        Forward pass of StreamDNN.

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