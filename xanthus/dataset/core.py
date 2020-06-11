"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import os
from typing import Optional, Any, Generator, Tuple, Callable
from collections import defaultdict

from scipy.sparse import coo_matrix, csr_matrix
from numpy import ndarray
from pandas import DataFrame

import numpy as np
from .encoder import DatasetEncoder

from .utils import sample_zeros, construct_coo_matrix


class Dataset:
    """
    A simple recommender dataset abstraction with utilities training and evaluating
    recommendation models.



    """

    def __init__(
        self,
        interactions: coo_matrix,
        user_meta: Optional[coo_matrix] = None,
        item_meta: Optional[coo_matrix] = None,
        encoder: DatasetEncoder = None,
    ) -> "Dataset":
        """

        Parameters
        ----------
        interactions
        user_meta
        item_meta
        encoder

        """

        self.encoder = encoder
        self.interactions = interactions
        self.user_meta = user_meta
        self.item_meta = item_meta

    @property
    def users(self) -> ndarray:
        """Get all users _with interactions_ in the dataset."""
        return np.unique(self.interactions.nonzero()[0])

    @property
    def items(self) -> ndarray:
        """Get all items _with interactions_ in the dataset."""
        return np.unique(self.interactions.nonzero()[1])

    @property
    def all_users(self):
        """Get all users in the dataset (inc. those with no interactions)."""
        return np.arange(self.interactions.shape[0])

    @property
    def all_items(self):
        """Get all items in the dataset (inc. those with no interactions)."""
        return np.arange(self.interactions.shape[1])

    @property
    def history(self):
        """Get the history (items a user has interacted with) of each user."""
        mat = self.interactions.tocsr()
        return [mat[i].nonzero()[1].tolist() for i in self.users]

    def iter(
        self,
        negative_samples: int = 0,
        output_dim: int = 1,
        shuffle: bool = True,
        aux_matrix: Optional[csr_matrix] = None,
    ) -> Generator:
        """
        Iterate over a sequence of ([user_vector, item_vector], ratings).

        Parameters
        ----------
        negative_samples
        output_dim
        shuffle
        aux_matrix

        Returns
        -------
        output: Generator

        """

        # must cast interactions to csr so we can use indexing on the matrix.
        interactions: csr_matrix = self.interactions.tocsr()

        # setup user metadata
        if self.user_meta is not None:
            user_meta = self.user_meta.tocsr()
        else:
            user_meta = None

        # setup item metadata
        if self.item_meta is not None:
            item_meta = self.item_meta.tocsr()
        else:
            item_meta = None

        users, items = interactions.nonzero()
        ratings = interactions.data

        if negative_samples > 0:
            # the aux_matrix should include additional interactions you wish to consider
            # _exclusively_ for the purposes of generating negative samples.
            if aux_matrix is not None:
                interactions += aux_matrix

            # run sampling.
            users, items, ratings = self._concatenate_negative_samples(
                users, items, ratings, interactions, negative_samples
            )

        # optionally shuffle the users, items and ratings.
        if shuffle:
            mask = np.arange(users.shape[0])
            np.random.shuffle(mask)
            users = users[mask]
            items = items[mask]
            ratings = ratings[mask]

        ratings.reshape(-1, 1)

        # stack user ids with associated user metadata.
        if user_meta is not None and output_dim > 1:
            users = self._iter_meta(users, user_meta, output_dim)
        else:
            users = users.reshape(-1, 1)

        # stack item ids with associated item metadata.
        if item_meta is not None and output_dim > 1:
            items = self._iter_meta(items, item_meta, output_dim)
        else:
            items = items.reshape(-1, 1)

        for (user, item, rating) in zip(users, items, ratings):
            yield user, item, rating

    def _concatenate_negative_samples(
        self, users: ndarray, items: ndarray, ratings: ndarray, mat: csr_matrix, n: int
    ) -> Tuple[ndarray, ...]:
        """

        Parameters
        ----------
        users
        items
        ratings
        mat
        n

        Returns
        -------

        """

        neg_users, neg_items, neg_ratings = self._sample_negatives(mat, n)

        users = np.concatenate((users, neg_users))
        items = np.concatenate((items, neg_items))
        ratings = np.concatenate((ratings, neg_ratings))

        return users, items, ratings

    def _sample_negatives(self, mat: csr_matrix, n: int) -> Tuple[ndarray, ...]:
        """

        Parameters
        ----------
        mat
        n

        Returns
        -------

        """

        data = np.asarray(list(sample_zeros(self.users, mat, n))).astype(np.int32)
        return data[:, 0], data[:, 1], np.zeros(shape=data.shape[0])

    @staticmethod
    def _iter_meta(ids: ndarray, meta: csr_matrix, n_dim: int) -> Generator:
        """

        Parameters
        ----------
        ids
        meta
        n_dim

        Returns
        -------

        """

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
        normalize: Optional[Callable[[ndarray], ndarray]] = None,
        encoder: Optional[DatasetEncoder] = None,
        **kwargs: Optional[Any]
    ) -> "Dataset":
        """

        Parameters
        ----------
        interactions
        user_meta
        item_meta
        normalize
        encoder
        kwargs

        Returns
        -------

        """

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

        if normalize is not None:
            # ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min())
            ratings = normalize(ratings)

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
        """

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """

        return tuple(map(np.asarray, zip(*self.iter(*args, **kwargs))))

    def to_txt(
        self, dirname: str, *args: Optional[Any], **kwargs: Optional[Any]
    ) -> None:
        """
        Dump the dataset to a set of .txt files.

        Parameters
        ----------
        dirname
        args
        kwargs

        Returns
        -------

        """

        os.makedirs(dirname, exist_ok=True)
        users = open(os.path.join(dirname, "user.txt"), "ab")
        items = open(os.path.join(dirname, "item.txt"), "ab")
        ratings = open(os.path.join(dirname, "ratings.txt"), "ab")

        for user, item, rating in self.iter(*args, **kwargs):
            np.savetxt(users, np.atleast_2d(user), fmt="%i", delimiter=" ")
            np.savetxt(items, np.atleast_2d(item), fmt="%i", delimiter=" ")
            np.savetxt(ratings, np.atleast_1d(rating))

        users.close()
        items.close()
        ratings.close()

    def __iter__(self) -> Generator:
        yield from self.iter()
