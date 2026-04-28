import torch
import torch.nn as nn

from tscls.models.base_dnn import DNNBase


class BaseTrainer:
    """
    Trainer for the reference DNN model (Model 1).
    This trainer fits the reference classifier on the non-drift data
    and extracts the latent representations from the last hidden layer
    after training.
    """

    def __init__(
        self,
        loss_function: nn.Module,
        gamma1: float,
        s1: int,
        device: str | None = None,
    ) -> None:
        """
        Initialize the BaseTrainer instance.

        Parameters
        ----------
        loss_function : nn.Module
            Loss function used for training the reference DNN model.
        gamma1 : float
            Hyperparameter used to control the learning rate update.
        s1 : int
            Step size at which the learning rate is updated.
        device : str | None, optional
            Device on which computations should be performed. If None,
            it will be determined automatically ("cuda" if available, otherwise "cpu").
        """

        if s1 <= 0:
            raise ValueError("s1 must be a positive integer.")
        if not (0.0 < gamma1 <= 1.0):
            raise ValueError("gamma1 must be in the interval (0, 1].")

        self.loss_function = loss_function
        self.gamma1 = gamma1
        self.s1 = s1
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _update_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Update the learning rate of the optimizer according to the gamma1 hyperparameter.

        This method is used to update the learning rate of the optimizer every s1 epochs.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be updated.
        """

        for param_group in optimizer.param_groups:
            param_group["lr"] *= self.gamma1

    def train_model(
        self,
        model: DNNBase,
        x_reference: torch.Tensor,
        y_reference: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        batch_size: int,
    ) -> list[float]:
        """
        Train the reference DNN model on the non-drift data.

        Parameters
        ----------
        model : DNNBase
            The reference DNN model to be trained.
        x_reference : torch.Tensor
            The input data for training the reference DNN model.
        y_reference : torch.Tensor
            The target data for training the reference DNN model.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for training.
        epochs : int
            The number of epochs to train the model.
        batch_size : int
            Number of samples per mini-batch.

        Returns
        -------
        list[float]
            A list of mean training losses, one value per epoch.
        """

        model.to(self.device)
        model.train()

        x_reference = x_reference.to(self.device)
        y_reference = y_reference.to(self.device).float().view(-1, 1)
        n = x_reference.shape[0]

        losses: list[float] = []

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            x_shuffled = x_reference[perm]
            y_shuffled = y_reference[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n, batch_size):
                x_batch = x_shuffled[start : start + batch_size]
                y_batch = y_shuffled[start : start + batch_size]

                optimizer.zero_grad()
                logits = model(x_batch)
                loss = self.loss_function(logits, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item())
                n_batches += 1

            losses.append(epoch_loss / n_batches)

            if (epoch + 1) % self.s1 == 0:
                self._update_learning_rate(optimizer)

        return losses

    def extract_latents(
        self,
        model: DNNBase,
        x_reference: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract latent representations from a trained DNNBase model.

        Parameters
        ----------
        model : DNNBase
            The model from which to extract latent representations.
        x_reference : torch.Tensor
            Input data for the model.

        Returns
        -------
        torch.Tensor
            Extracted latent representations.
        """
        model.to(self.device)
        model.eval()

        x_reference = x_reference.to(self.device)

        with torch.no_grad():
            _, latents = model.forward_with_latent(x_reference)

        return latents.cpu()