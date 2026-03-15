import torch
import torch.nn as nn

from tscls.models.base_dnn import DNNBase


class BaseTrainer:
    """
    Trainer for the reference DNN model.
    This trainer fits the reference classifier on the non-drift data
    and extracts the latent representations from the last hidden layer
    after training.
    """

    def __init__(
        self,
        loss_function: nn.Module,
        device: str | None = None,
    ) -> None:
        """
        Initialize a BaseTrainer instance.

        Parameters
        ----------
        loss_function : nn.Module
            Loss function used for training.
        device : str | None, optional
            Device on which computations should be performed. If None,
            it will be determined automatically ("cuda" if available, otherwise "cpu").
            Defaults to None.
        """

        self.loss_function = loss_function
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def train_model(
        self,
        model: DNNBase,
        x_reference: torch.Tensor,
        y_reference: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epochs: int,
    ) -> list[float]:
        """
        Train a DNNBase instance on the non-drift data.

        Parameters
        ----------
        model : DNNBase
            The model to be trained.
        x_reference : torch.Tensor
            The input data for the non-drift class.
        y_reference : torch.Tensor
            The target data for the non-drift class.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training.
        epochs : int
            The number of epochs to train the model.

        Returns
        -------
        list[float]
            A list of the training losses at each epoch.
        """

        model.to(self.device)
        model.train()

        x_reference = x_reference.to(self.device)
        y_reference = y_reference.to(self.device).float()
        losses = []

        for _ in range(epochs):
            optimizer.zero_grad()

            logits = model(x_reference)
            loss = self.loss_function(logits, y_reference)

            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        return losses

    def extract_latents(
        self,
        model: DNNBase,
        x_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract latent representations from a DNNBase model.

        Parameters
        ----------
        model : DNNBase
            The model from which to extract the latent representations.
        x_reference : torch.Tensor
            Input data for the model.

        Returns
        -------
        torch.Tensor
            The extracted latent representations.
        """

        model.to(self.device)
        model.eval()

        x_reference = x_reference.to(self.device)

        with torch.no_grad():
            _, latents = model.forward_with_latent(x_reference)

        return latents.cpu()