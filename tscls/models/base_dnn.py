from typing import List, Tuple

import torch
import torch.nn as nn


class DNNBase(nn.Module):
    """
    A fully connected feedforward deep neural network.
    This model is used as the base classifier in the drift
    detection pipeline.

    The output of the last hidden layer is returned as a latent
    representation, which is used by the autoencoder to detect potential drift.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
    ) -> None:
        """
        Initialize the DNNBase instance.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : List[int]
            List of hidden layer dimensions.
        output_dim : int
            Number of output features.
        """
        
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, output_dim)
        self.activation = nn.ReLU()

    def forward_with_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DNNBase instance returning both
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

        h = x
        for layer in self.hidden_layers:
            h = self.activation(layer(h))

        latent = h
        logits = self.output_layer(latent)

        return logits, latent

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DNNBase instance.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The classifier output.
        """

        logits, _ = self.forward_with_latent(x)
        return logits