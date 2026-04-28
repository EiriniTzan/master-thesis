import numpy as np

from tscls.datasets.base import SyntheticDataset


class SEA(SyntheticDataset):
    """
    SEA concepts dataset generator.

    Each sample has three features uniformly drawn from [0, 10].
    The binary label is 1 when the sum of the first two features is
    at most a concept-specific threshold theta:

        function_id 0 → theta = 8.0
        function_id 1 → theta = 9.0
        function_id 2 → theta = 7.0
        function_id 3 → theta = 9.5

    Parameters
    ----------
    function_id : int
        Concept index (0–3) that selects the decision threshold.
    balanced : bool, optional
        When True (default) the two classes are forced to have exactly
        n_samples // 2 members via rejection sampling.

    Raises
    ------
    ValueError
        If function_id is not in {0, 1, 2, 3}.
    """

    _THRESHOLDS: dict[int, float] = {0: 8.0, 1: 9.0, 2: 7.0, 3: 9.5}

    def __init__(self, function_id: int, balanced: bool = True) -> None:
        if function_id not in self._THRESHOLDS:
            raise ValueError(
                f"function_id must be one of "
                f"{list(self._THRESHOLDS.keys())}, got {function_id}."
            )
        self._function_id = function_id
        self._balanced = balanced

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_features(self) -> int:
        return 3

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def function_id(self) -> int:
        """Concept index (0–3)."""
        return self._function_id

    @property
    def threshold(self) -> float:
        """Decision boundary used by this concept."""
        return self._THRESHOLDS[self._function_id]


    def generate(
        self,
        n_samples: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a labelled SEA dataset for this concept.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate. Must be even when the
            instance was created with balanced=True.
        seed : int | None, optional
            Seed for the random number generator.

        Returns
        -------
        X : np.ndarray of shape (n_samples, 3), dtype float32
            Feature matrix.
        y : np.ndarray of shape (n_samples,), dtype float32
            Binary labels (0.0 or 1.0).
        """

        rng = np.random.default_rng(seed)

        if not self._balanced:
            X = rng.uniform(0, 10, size=(n_samples, 3)).astype(np.float32)
            y = (X[:, 0] + X[:, 1] <= self.threshold).astype(np.float32)
            return X, y

        target = n_samples // 2
        X_list: list[np.ndarray] = []
        y_list: list[float] = []
        n_pos = n_neg = 0

        while n_pos < target or n_neg < target:
            x = rng.uniform(0, 10, size=3).astype(np.float32)
            label = float(x[0] + x[1] <= self.threshold)
            if label == 1.0 and n_pos < target:
                X_list.append(x)
                y_list.append(1.0)
                n_pos += 1
            elif label == 0.0 and n_neg < target:
                X_list.append(x)
                y_list.append(0.0)
                n_neg += 1

        idx = rng.permutation(n_samples)
        X_arr = np.array(X_list, dtype=np.float32)[idx]
        y_arr = np.array(y_list, dtype=np.float32)[idx]
        return X_arr, y_arr
