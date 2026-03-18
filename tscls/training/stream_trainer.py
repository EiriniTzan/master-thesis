import torch
import torch.nn as nn

from tscls.core.sample import Sample
from tscls.models.stream_dnn import StreamDNN


class StreamTrainer:
    """
    Trainer for the adaptive DNN model (Model 2).
    This trainer performs sample-by-sample online updates on streaming data
    and implements learning rate adaptation through gamma2 and s2.
    """

    def __init__(
        self,
        loss_function: nn.Module,
        gamma2: float,
        s2: int,
        device: str | None = None,
    ) -> None:
        """
        Initialize the StreamTrainer instance.

        Parameters
        ----------
        loss_function : nn.Module
            Loss function used for online training of the adaptive DNN model.
        gamma2 : float
            Hyperparameter used to control the learning rate adaptation.
        s2 : int
            Step size at which the learning rate is adapted.
        device : str | None, optional
            Device on which computations should be performed. If None,
            it will be determined automatically ("cuda" if available, otherwise "cpu").
        """
        
        if s2 <= 0:
            raise ValueError("s2 must be a positive integer.")
        if not (0.0 < gamma2 <= 1.0):
            raise ValueError("gamma2 must be in the interval (0, 1].")

        self.loss_function = loss_function
        self.gamma2 = gamma2
        self.s2 = s2
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.step_count = 0

    def _update_learning_rate(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Update the learning rate of the optimizer according to the gamma2 hyperparameter.

        This method is used to update the learning rate of the optimizer every s2 steps.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer whose learning rate should be updated.
        """

        for param_group in optimizer.param_groups:
            param_group["lr"] *= self.gamma2

    def train_on_sample(
        self,
        model: StreamDNN,
        sample: Sample,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, torch.Tensor, torch.Tensor]:
        """
        Perform an online supervised update of the StreamDNN model on a single sample.

        Parameters
        ----------
        model : StreamDNN
            The StreamDNN model to be updated.
        sample : Sample
            The sample to be used for updating the model.
        optimizer : torch.optim.Optimizer
            The optimizer to be used for updating the model.

        Returns
        -------
        tuple[float, torch.Tensor, torch.Tensor]
            A tuple containing the loss, the predicted logits and the latent representation of the sample.
        """
        
        if sample.y is None:
            raise ValueError("Online supervised update requires an available label.")

        model.to(self.device)
        model.train()

        x = sample.x.to(self.device).unsqueeze(0)
        y = torch.tensor([[sample.y]], dtype=torch.float32, device=self.device)

        optimizer.zero_grad()

        logits, latent = model.forward_with_latent(x)
        loss = self.loss_function(logits, y)

        loss.backward()
        optimizer.step()

        self.step_count += 1
        if self.step_count % self.s2 == 0:
            self._update_learning_rate(optimizer)

        return float(loss.item()), logits.detach().cpu(), latent.detach().cpu()
