import time
import numpy as np
from scipy.sparse import coo_matrix, dok_matrix
from itertools import islice
from collections import defaultdict
from .encoder import DatasetEncoder


def construct_coo_matrix(p, q, r=None, dtype="float32"):
    r = r if r is not None else np.ones_like(p)
    return coo_matrix((r.astype(dtype), (p, q)))


def rejection_sample(j, sampled=None):
    sampled = set(sampled) if sampled is not None else set()

    while True:
        choice = np.random.randint(j)
        if choice not in sampled:
            yield choice
            sampled.add(choice)


def single_sample_zeros(i, mat, k):
    yield from islice(rejection_sample(mat.shape[1], mat[i].nonzero()[1]), k)


def sample_zeros(ids, mat, k):
    yield from ((i, z) for i in ids for z in single_sample_zeros(i, mat, k))


class Dataset:
    def __init__(self, interactions, user_meta=None, item_meta=None, encoder=None):
        self.encoder = encoder
        self.interactions = interactions
        self.user_meta = user_meta
        self.item_meta = item_meta

    def __iter__(self):
        yield from self.iter()

    def to_arrays(self, *args, **kwargs):
        # This is a little grim.
        arr = [[a, b, c] for a, b, c in self.iter(*args, **kwargs)]
        return (
            np.asarray([a for a, _, _ in arr]),
            np.asarray([b for _, b, _ in arr]),
            np.asarray([c for _, _, c in arr]),
        )

    def iter(self, negative_samples=0, output_dim=1):
        interactions = self.interactions.tocsr()

        if self.user_meta is not None:
            user_meta = self.user_meta.tocsr()
        else:
            user_meta = None

        if self.item_meta is not None:
            item_meta = self.item_meta.tocsr()
        else:
            item_meta = None

        users, items = interactions.nonzero()
        ratings = interactions.data

        print(len(ratings))
        if negative_samples > 0:
            neg_users, neg_items, neg_ratings = self._sample_negatives(interactions,
                                                                       negative_samples)

            users = np.concatenate((users, neg_users))
            items = np.concatenate((items, neg_items))
            ratings = np.concatenate((ratings, neg_ratings))

            np.random.shuffle(users)
            np.random.shuffle(items)
            np.random.shuffle(ratings)

        print(len(ratings))
        ratings.reshape(-1, 1)

        if user_meta is not None and output_dim > 1:
            users = self._iter_meta(users, user_meta, output_dim)
        else:
            users = users.reshape(-1, 1)

        if item_meta is not None and output_dim > 1:
            items = self._iter_meta(items, item_meta, output_dim)
        else:
            items = items.reshape(-1, 1)

        for (user, item, rating) in zip(users, items, ratings):
            yield user, item, rating

    @staticmethod
    def _sample_negatives(mat, n):
        users = mat.nonzero()[0]
        data = np.asarray(list(sample_zeros(users, mat, n))).astype(np.int32)
        return data[:, 0], data[:, 1], np.zeros(shape=data.shape[0])

    @staticmethod
    def _iter_meta(ids, meta, n_dim):
        groups = defaultdict(list)
        _ids, tags = meta.nonzero()

        for _id, _tag in zip(_ids, tags):
            groups[_id].append(_tag)

        for _id in ids:
            group = groups[_id]
            padding = [0] * max(0, n_dim - len(group))
            features = [_id, *group, *padding][:n_dim]
            yield features

    @classmethod
    def from_frame(
        cls,
        interactions,
        user_meta=None,
        item_meta=None,
        normalize=True,
        encoder=None,
        **kwargs
    ):

        users = set(interactions["user"].tolist())
        items = set(interactions["item"].tolist())

        if user_meta is not None:
            users.update(set(user_meta["user"].tolist()))

        if item_meta is not None:
            items.update(set(item_meta["item"].tolist()))

        if encoder is None:
            encoder = DatasetEncoder(**kwargs)
            encoder.fit(
                users,
                items,
                user_meta["tag"].unique() if user_meta is not None else None,
                item_meta["tag"].unique() if item_meta is not None else None,
            )

        encoded = encoder.transform(interactions["user"], interactions["item"])

        if "rating" not in interactions.columns:
            ratings = np.ones_like(encoded["users"]).astype(np.float32)
        else:
            ratings = interactions["rating"].values.astype(np.float32)

        if normalize and not np.unique(ratings).shape[0] < 2:
            ratings = 1.0 / (np.exp(ratings) - 1.0)

        interactions = construct_coo_matrix(encoded["users"], encoded["items"], ratings)

        if user_meta is not None:
            encoded = encoder.transform(
                users=user_meta["user"], user_features=user_meta["tag"]
            )
            user_meta = construct_coo_matrix(encoded["users"], encoded["user_features"])

        if item_meta is not None:
            encoded = encoder.transform(
                items=item_meta["item"], item_features=item_meta["tag"]
            )
            item_meta = construct_coo_matrix(encoded["items"], encoded["item_features"])

        return cls(
            interactions, user_meta=user_meta, item_meta=item_meta, encoder=encoder
        )
