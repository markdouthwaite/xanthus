import os
from typing import Optional, Any, Generator, Tuple
from collections import defaultdict

from scipy.sparse import coo_matrix, csr_matrix
from numpy import ndarray
from pandas import DataFrame

import numpy as np
from .encoder import DatasetEncoder

from .utils import sample_zeros, construct_coo_matrix


class Dataset:
    def __init__(
        self,
        interactions: coo_matrix,
        user_meta: Optional[coo_matrix] = None,
        item_meta: Optional[coo_matrix] = None,
        encoder: DatasetEncoder = None,
    ) -> "Dataset":
        self.encoder = encoder
        self.interactions = interactions
        self.user_meta = user_meta
        self.item_meta = item_meta

    def iter(
        self, negative_samples: int = 0, output_dim: int = 1, shuffle: bool = True
    ) -> Generator:

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

        if negative_samples > 0:
            users, items, ratings = self._concatenate_negative_samples(
                users, items, ratings, interactions, negative_samples
            )

        if shuffle:
            mask = np.arange(users.shape[0])
            np.random.shuffle(mask)
            users = users[mask]
            items = items[mask]
            ratings = ratings[mask]

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

    def _concatenate_negative_samples(
        self, users: ndarray, items: ndarray, ratings: ndarray, mat: csr_matrix, n: int
    ) -> Tuple[ndarray, ...]:
        neg_users, neg_items, neg_ratings = self._sample_negatives(mat, n)

        users = np.concatenate((users, neg_users))
        items = np.concatenate((items, neg_items))
        ratings = np.concatenate((ratings, neg_ratings))

        return users, items, ratings

    @staticmethod
    def _sample_negatives(mat: csr_matrix, n: int) -> Tuple[ndarray, ...]:
        users = mat.nonzero()[0]
        data = np.asarray(list(sample_zeros(users, mat, n))).astype(np.int32)
        return data[:, 0], data[:, 1], np.zeros(shape=data.shape[0])

    @staticmethod
    def _iter_meta(ids: ndarray, meta: csr_matrix, n_dim: int) -> Generator:
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
        interactions: DataFrame,
        user_meta: Optional[DataFrame] = None,
        item_meta: Optional[DataFrame] = None,
        normalize: bool = True,
        encoder: Optional[DatasetEncoder] = None,
        **kwargs: Optional[Any]
    ) -> "Dataset":

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

        interactions_shape = (
            len(encoder.user_mapping) + 1,
            len(encoder.item_mapping) + 1,
        )

        interactions = construct_coo_matrix(
            encoded["users"], encoded["items"], ratings, shape=interactions_shape
        )

        if user_meta is not None:
            user_meta_shape = (
                len(encoder.user_mapping) + 1,
                len(encoder.user_tag_mapping) + 1,
            )
            encoded = encoder.transform(
                users=user_meta["user"], user_tags=user_meta["tag"]
            )
            user_meta = construct_coo_matrix(
                encoded["users"], encoded["user_features"], shape=user_meta_shape
            )

        if item_meta is not None:
            item_meta_shape = (
                len(encoder.item_mapping) + 1,
                len(encoder.item_tag_mapping) + 1,
            )
            encoded = encoder.transform(
                items=item_meta["item"], item_tags=item_meta["tag"]
            )
            item_meta = construct_coo_matrix(
                encoded["items"], encoded["item_features"], shape=item_meta_shape
            )

        return cls(
            interactions, user_meta=user_meta, item_meta=item_meta, encoder=encoder
        )

    def to_arrays(
        self, *args: Optional[Any], **kwargs: Optional[Any]
    ) -> Tuple[ndarray, ...]:
        return tuple(map(np.asarray, zip(*self.iter(*args, **kwargs))))

    def to_txt(self, dir: str, *args: Optional[Any], **kwargs: Optional[Any]) -> None:
        os.makedirs(dir, exist_ok=True)
        users = open(os.path.join(dir, "user.txt"), "ab")
        items = open(os.path.join(dir, "item.txt"), "ab")
        ratings = open(os.path.join(dir, "ratings.txt"), "ab")

        for user, item, rating in self.iter(*args, **kwargs):
            np.savetxt(users, np.atleast_2d(user), fmt="%i", delimiter=" ")
            np.savetxt(items, np.atleast_2d(item), fmt="%i", delimiter=" ")
            np.savetxt(ratings, np.atleast_1d(rating))
        users.close()
        items.close()
        ratings.close()

    def __iter__(self) -> Generator:
        yield from self.iter()
