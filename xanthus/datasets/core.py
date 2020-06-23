"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Any, List, Tuple, Callable, Iterator
from collections import defaultdict

from scipy.sparse import coo_matrix, csr_matrix
from numpy import ndarray
from pandas import DataFrame

import numpy as np
from .encoder import DatasetEncoder


from .utils import construct_coo_matrix, sample_negatives, SamplerCallable


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
        sampler: SamplerCallable = sample_negatives,
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
        self.sampler = sampler

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
    def all_users(self) -> ndarray:
        """Get all users in the dataset (inc. those with no interactions)."""

        return np.arange(self.interactions.shape[0])

    @property
    def all_items(self) -> ndarray:
        """Get all items in the dataset (inc. those with no interactions)."""

        return np.arange(self.interactions.shape[1])

    @property
    def history(self) -> ndarray:
        """Get the history (items a user has interacted with) of each user."""

        mat = self.interactions.tocsr()
        g = (mat[i].nonzero()[1].tolist() for i in self.users)
        return np.fromiter(g, dtype=int)

    def iter(
        self,
        negative_samples: int = 0,
        output_dim: int = 1,
        shuffle: bool = True,
        aux_matrix: Optional[coo_matrix] = None,
        sampling_mode: str = "relative",
    ) -> Iterator[Tuple[ndarray, ndarray, float]]:
        """
        Iterate over a sequence of ([user_vector_{i}, item_vector_{i}], ratings_{i}).

        In practice, this will result in each user-item interaction being yielded,
        optionally with additional metadata, and optionally with 'n' negative sample
        instances.

        By default (with 'output_dim=1', and/or with no metadata), this will yield:

        ([user_{i}], [item_{i}}], ratings_{i})

        Concretely, this may be: ([0], [1], 1) If user/item meta-data is provided, this
        will be lazily injected into the yielded value, for example:

        ([user_{i}, user_tag_{i}{1}, ..., user_tag_{i}{n}],
         [item_{i}, item_tag_{i}{1}, ..., item_tag_{i}{n}],
         ratings_{i})

        Again, concretely: ([0, 21, 82], [1, 97, 64], 1). Make sure to set 'output_dim'
        if you want your metadata rendered (if you provided it)!

        Parameters
        ----------
        negative_samples: int
            The total number of negative samples (for each positive sample) you wish
            to take from the provided interactions set (and auxiliary matrix, if
            provided).
        output_dim: int
            The output dimensions for _both_ the encoded user- and item-vectors. Note
            that this will only be applied if user/item metadata is provided, otherwise
            all output vectors will have 'output_dim=1'.
        shuffle: int
            Indicate whether the output data should be shuffled.
        aux_matrix: coo_matrix, optional
            Provide a sparse matrix of the same shape as the 'interactions' matrix with
            additional interactions terms. These terms will be used when taking negative
            samples. This can be useful if this Dataset is a 'test' dataset, and you
            wish to draw negative samples from items a user has never interacted with
            when generating an evaluation set. This is the process described in [1].
        sampling_mode: str
            If negative sampling is used, specify the sampling mode you wish to use.
            See 'xanthus.dataset.utils.single_negative_sample' for more details.

        Returns
        -------
        output: Generator
            A generator yielding user/item vectors and the associated pairing's rating.

        See Also
        --------
            xanthus.evaluate.utils.he_sample
            xanthus.dataset.utils.single_negative_sample

        References
        ----------
        [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569

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
            users, items, ratings = self.sampler(
                users,
                items,
                ratings,
                interactions,
                negative_samples,
                sampling_mode,
                aux_matrix,
                concat=True,
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

    @staticmethod
    def _iter_meta(ids: ndarray, meta: csr_matrix, n_dim: int) -> Iterator[List[int]]:
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
    def from_df(
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
        """Transform `iter` output into a set of arrays."""

        return tuple(map(np.asarray, zip(*self.iter(*args, **kwargs))))

    def __iter__(self) -> Iterator[Tuple[ndarray, ndarray, float]]:
        """Iterate over the dataset."""

        yield from self.iter()
