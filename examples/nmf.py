import timeit
import random
import numpy as np
from numpy import ndarray
from itertools import islice
from scipy.sparse import coo_matrix

from lightfm.data import Dataset


class Dataset:
    def __init__(self):
        self._user_mapping = {}
        self._item_mapping = {}
        self._user_feature_mapping = {}
        self._item_feature_mapping = {}

    def fit(self, x, user_features=None, item_features=None):
        pass

    def partial_fit(self, x, user_features=None, item_features=None):
        self._user_mapping = self._fit_mapping(x[:, 0], self._user_mapping)
        self._item_mapping = self._fit_mapping(x[:, 1], self._item_mapping)

        if user_features is not None:
            self._user_feature_mapping = self._fit_feature_mapping(
                user_features, self._user_feature_mapping, self._user_mapping
            )
        if item_features is not None:
            self._item_feature_mapping = self._fit_feature_mapping(
                item_features, self._item_feature_mapping, self._item_mapping
            )

    def _fit_feature_mapping(self, data, encodings, aux_encodings):
        encodings = self._fit_mapping(data, encodings, offset=len(aux_encodings),)
        encodings.update(aux_encodings)
        return encodings

    @staticmethod
    def _fit_mapping(data, encodings, offset=0):
        for e in data:
            encodings.setdefault(e, len(encodings) + offset)
        return encodings

    def inverse_transform(self, x, user_features=None, item_features=None):
        pass

    def transform(self):
        pass

    def fit_transform(self):
        pass


def construct_coo_matrix(p, q, r=None, dtype="float32"):
    r = r if r is not None else np.ones_like(p)
    return coo_matrix((r.astype(dtype), (p, q)))


def sample_interactions(n, m, k):
    u = np.random.randint(n, size=k)
    i = np.random.randint(m, size=k)
    return np.c_[u, i]


def rejection_sample(j, sampled=None):
    sampled = set(sampled) if sampled is not None else set()

    while True:
        choice = np.random.randint(j)
        if choice not in sampled:
            yield choice
            sampled.add(choice)


def sample_zeros(i, mat, k):
    return np.asarray(
        list(islice(rejection_sample(mat.shape[1], mat[i].nonzero()[1]), k))
    )


np.random.seed(42)

# x = sample_interactions(100, 200, 1000)
# c = construct_coo_matrix(x[:, 0], x[:, 1]).tocsr()
#
# print(c[0, 8])
#
# print(c.shape[1])
# print(c[0].nonzero()[1])
#
# print(sample_zeros(0, c, 6))
