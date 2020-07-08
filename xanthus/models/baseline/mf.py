"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Any, List
from itertools import islice

from numpy import asarray

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking

from xanthus.datasets import Dataset


class MatrixFactorizationModel:
    """
    A simple adapter for 'non-neural' matrix factorization algorithms.

    This class wraps the functionality provided by the Implicit library [1] which is
    itself based (partly) upon [2]. It is provided as a baseline model, along with
     the simple 'PopRankModel'.

    References
    ----------
    [1] https://github.com/benfred/implicit/tree/master/implicit
    [2] https://www.comp.nus.edu.sg/~xiangnan/papers/sigir16-eals-cm.pdf

    See Also
    --------
    xanthus.models.baseline.PopRankModel

    """

    _methods = {
        "als": AlternatingLeastSquares,
        "bpr": BayesianPersonalizedRanking,
    }

    def __init__(self, method: str = "als", **kwargs: Optional[Any]) -> None:
        """Initialise a MatrixFactorizationModel."""

        self._model = self._methods[method](**kwargs)
        self._mat = None

    def fit(self, dataset: Dataset) -> "MatrixFactorizationModel":
        """
        Fit the model to a provided Dataset.

        Parameters
        ----------
        dataset: Dataset
            An input dataset.

        Returns
        -------
        output: MatrixFactorizationModel
            Returns itself. How fun.

        See Also
        --------
        xanthus.datasets.Dataset

        """

        self._mat = dataset.interactions.tocsr()
        self._model.fit(self._mat.T)
        return self

    def predict(
        self,
        dataset: Dataset,
        users: Optional[List[int]] = None,
        items: Optional[List[int]] = None,
        n: int = 3,
        **kwargs: Optional[Any]
    ) -> List[List[int]]:
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
        Returns
        -------
        output: list
            A list, where each element corresponds to a list of recommendations.

        """

        users = users if users is not None else dataset.users
        users = users.flatten()

        user_items = dataset.interactions.tocsr()

        if items is None:
            rec = self._model.recommend
            recommended = asarray(
                [[_[0] for _ in rec(user, user_items, N=n, **kwargs)] for user in users]
            )

        else:
            user_factors = self._model.user_factors
            item_factors = self._model.item_factors
            recommended = []
            for i, user in enumerate(users):
                user_factor = user_factors[user]
                user_items = items[i].flatten()
                scores = item_factors.dot(user_factor)
                user_recs = islice(
                    sorted(zip(user_items, scores[user_items]), key=lambda _: -_[1]), n
                )
                recommended.append([_[0] for _ in user_recs])

        return recommended
