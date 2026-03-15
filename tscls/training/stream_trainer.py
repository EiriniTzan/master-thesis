import torch
import torch.nn as nn

from tscls.models.stream_dnn import StreamDNN


class StreamTrainer:
    """
    Trainer for the online stream model (Model2).
    This class performs one online update step on the stream model
    during the adaptation phase of this model. The implementation is intended for
    sample-by-sample updates, following the online setting described in the paper.
    """

    def __init__(
        self,
        loss_function: nn.Module,
        device: str | None = None,
    ) -> None:
        """
        Initialize a StreamTrainer instance.

        Parameters
        ----------
        loss_function : nn.Module
            Loss function used for training the stream model.
        device : str | None, optional
            Device on which computations should be performed. If None,
            it will be determined automatically ("cuda" if available, otherwise "cpu").
        """

        self.loss_function = loss_function
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def update(
        self,
        model: StreamDNN,
        x_stream: torch.Tensor,
        y_stream: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Perform one online update step on the stream model during the adaptation phase.

        Parameters
        ----------
        model : StreamDNN
            The stream model to be updated.
        x_stream : torch.Tensor
            The input data point from the stream.
        y_stream : torch.Tensor
            The target data point from the stream.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training.

        Returns
        -------
        tuple[float, torch.Tensor, torch.Tensor]
            A tuple containing the training loss, the classifier output and the last
            hidden layer activation (latent representation of the input sample).
        """

        model.to(self.device)
        model.train()

        x_stream = x_stream.to(self.device)
        y_stream = y_stream.to(self.device).float()

        optimizer.zero_grad()

        logits, latent = model.forward_with_latent(x_stream)
        loss = self.loss_function(logits, y_stream)

        loss.backward()
        optimizer.step()

        return float(loss.item()), logits.detach().cpu(), latent.detach().cpu()
