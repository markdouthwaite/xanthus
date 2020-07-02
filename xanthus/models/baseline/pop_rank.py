from typing import Optional, Dict, Any
import numpy as np
from numpy import ndarray

from xanthus.datasets import Dataset
from xanthus.models import base


class PopRankModel(base.RecommenderModel):
    """
    A Popularity Ranking recommender model. This class implements some simple
    popularity-based recommendation approaches. By default, it will simply return
    the most popular items that meet a given criteria (e.g. in a given list). It
    also implements a type of weighted shuffle to inject lower-ranked items higher
    in the popularity rankings.

    Parameters
    ----------
    alpha: float
        The 'strength' of weighting towards popular terms: '0' would correspond to
        a uniform shuffle. Only used when sampling="weighted".
        Implemented as in [1].
    limit: int
        The maximum number of items to select for ordering. Increasing this will
        impact performance for large sets of items.
    sampling: str
        The sampling method to use when generating recommendations. This is used to
        shuffle the popularity-ranked items (i.e. the popularity ranked items will be
        reordered according to this sampling method).
            * uniform - randomly shuffle top ranked items.
            * weighted - shuffle items, but tend to keep top ranked items higher in the
                         recommender's outputs.
            * None - well, it seems you don't want to shuffle the items at all! This
                     is 'pure' PopRank.

    References
    ----------
    [1] https://godatadriven.com/blog/elitist-shuffle-for-recommendation-systems/

    """

    def __init__(
        self, alpha: float = 1.0, limit: int = 100, sampling: Optional[str] = "uniform"
    ) -> None:
        """Time to initialise a PopRankModel."""

        self._sampling = sampling
        self._ranked = None
        self._alpha = alpha
        self._limit = limit

    def fit(self, dataset: Dataset) -> "PopRankModel":
        """
        Fit the model to a provided Dataset.

        Parameters
        ----------
        dataset: Dataset
            An input dataset.

        Returns
        -------
        output: PopRankModel
            Returns itself. How fun.

        See Also
        --------
        xanthus.datasets.Dataset

        """

        init_weights = np.asarray(dataset.interactions.sum(axis=0)).flatten()
        ranked = init_weights.argsort()[::-1]
        self._ranked = ranked
        return self

    def predict(
        self,
        dataset: Optional[Dataset] = None,
        users: ndarray = None,
        items: ndarray = None,
        excluded: Optional[ndarray] = None,
        n: int = 3,
    ) -> ndarray:
        """
        Generate predictions (recommendations) from the model. For each provided user,
        the output will be a set of items ranked in order of predicted preference.

        Parameters
        ----------
        dataset: Dataset, optional
            The dataset for which you wish to generate recommendations. If this is
            provided, this object's `all_items` and `users` will be used for the
            purpose of generating recommendations.
        users: list, optional
            An optional array of users for whom you wish to generate recommendations.
        items: list, optional
            An optional array of items you wish to be used in recommendations. This
            may be a subset of items for the purposes of ranking specific subsets of
            items, for example. Maybe you want to see which Star Wars movies are the
            most loved according to the model, for example.
        n: int
            The number of recommendations to be generated per user.
        excluded: list, optional
            An optional array of items to exclude from recommendations.

        Returns
        -------
        output: list
            A list, where each element corresponds to a list of recommendations.

        """

        if excluded is not None:
            valid = np.isin(excluded, self._ranked)
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

        # truncate items for filter/shuffle speed.
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
            if self._sampling is None:
                recommended = items[:n]
            else:
                recommended[i] += np.random.choice(
                    items, size=n, replace=False, p=weights
                )

        return recommended
