"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from itertools import islice

from numpy import asarray

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking


class MatrixFactorizationModel:
    """
    A simple adapter for 'non-neural' matrix factorization algorithms.

    This class wraps the functionality provided by the Implicit library [1] which is
    itself based (partly) upon [2].

    References
    ----------
    [1]
    [2] https://www.comp.nus.edu.sg/~xiangnan/papers/sigir16-eals-cm.pdf

    """

    _methods = {
        "als": AlternatingLeastSquares,
        "bpr": BayesianPersonalizedRanking,
    }

    def __init__(self, method: str = "als", **kwargs) -> None:
        self._model = self._methods[method](**kwargs)
        self._mat = None

    def fit(self, dataset):
        self._mat = dataset.interactions.tocsr()
        self._model.fit(self._mat.T)
        return self

    def predict(self, dataset, users=None, items=None, n=6, **kwargs):
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
