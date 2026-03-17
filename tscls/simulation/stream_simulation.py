import torch


class StreamSimulator:
    """
    Simulates a data stream through sample-by-sample generation.
    Each sample is provided individually, modeling an online setting in
    which data is received progressively over time, not in batch form.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> None:
        """
        Initialize a StreamSimulator instance.

        Parameters
        ----------
        data : torch.Tensor
            A tensor containing the data points of the stream.
        labels : torch.Tensor | None, optional
            A tensor containing the labels of the stream. If None, no labels are provided.
        """

        self.data = data
        self.labels = labels
        self.num_samples = data.shape[0]
        self.current_index = 0

    def has_next_sample(self) -> bool:
        """
        Checks if there are more samples in the stream.

        Returns
        -------
        bool
            True if there are more samples, False otherwise.
        """
        
        return self.current_index < self.num_samples

    def next_sample(self) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Returns the next sample from the stream.

        Returns a tuple containing the input tensor (x), the label tensor (y)
        and the index of the sample in the stream (i). If the stream contains
        no labels, y is None.

        Raises StopIteration if there are no more samples in the stream.
        """

        if not self.has_next_sample():
            raise StopIteration("No more samples in the stream.")

        i = self.current_index
        x = self.data[i]
        y = self.labels[i] if self.labels is not None else None

        self.current_index += 1

        return x, y, i

    def reset(self) -> None:
        """
        Resets the simulator to its initial state.

        This method resets the current index to zero, so that the next call to
        next_sample() will return the first sample from the stream.
        """

        self.current_index = 0