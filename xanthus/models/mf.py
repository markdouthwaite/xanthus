from numpy import asarray

from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking


class MatrixFactorizationModel:

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

    def predict(self, users, excluded_items=None, n=6, mat=None):
        mat = mat or self._mat
        rec = self._model.recommend
        return asarray(
            [
                [_[0] for _ in rec(user, mat, filter_items=excluded_items, N=n)]
                for user in users
            ]
        )
