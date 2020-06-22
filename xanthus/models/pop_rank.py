from typing import Optional, Dict, Any
import numpy as np
from numpy import ndarray

from xanthus.dataset import Dataset


class PopRankModel:
    def __init__(
        self, alpha: float = 1.0, limit: int = 100, sampling="uniform"
    ) -> None:
        """

        Parameters
        ----------
        alpha
        limit
        sampling

        References
        ----------
        [1] - https://godatadriven.com/blog/elitist-shuffle-for-recommendation-systems/

        """

        self._sampling = sampling
        self._ranked = None
        self._alpha = alpha
        self._limit = limit

    def fit(self, dataset: Dataset) -> "PopRankModel":
        init_weights = np.asarray(dataset.interactions.sum(axis=0)).flatten()
        ranked = init_weights.argsort()[::-1]
        self._ranked = ranked
        return self

    def predict(
        self,
        _: Dataset,
        users: ndarray,
        items: Optional[ndarray] = None,
        exclude_items: Optional[ndarray] = None,
        n: int = 3,
    ) -> ndarray:

        if exclude_items is not None:
            valid = np.isin(exclude_items, self._ranked)
            items = self._ranked[valid]

            if len(items) < n:
                raise ValueError(
                    f"Insufficient valid items available ({len(items)}, required {n})"
                    "after filtering excluded items."
                )

        elif items is not None:
            items = self._ranked[np.isin(items, self._ranked)]

        else:
            items = self._ranked

        items = items[: self._limit]

        if self._sampling == "weighted":
            weights = np.linspace(1, 0, num=len(items), endpoint=False)
            weights = np.power(weights, self._alpha)
            weights /= np.linalg.norm(weights, ord=1)
        else:
            weights = np.ones(len(items), dtype=np.float32)
            weights /= weights.sum()

        recommended = np.zeros((len(users), n), dtype=np.int32)
        for i, user in enumerate(users):
            recommended[i] += np.random.choice(items, size=n, replace=False, p=weights)

        return recommended

    def get_params(self) -> Dict[str, Any]:
        return {
            "ranked": self._ranked.tolist() if self._ranked is not None else None,
            "alpha": self._alpha,
            "limit": self._limit,
        }

    def set_params(self, params):
        pass
