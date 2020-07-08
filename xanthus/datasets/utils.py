"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Union, List, Set, Iterator, Tuple, Any, Callable
from itertools import islice

from pandas import DataFrame
from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
import numpy as np
from numpy import random, ones_like, ndarray, zeros


# this is a bit of a mess.
SamplerCallable = Callable[
    [ndarray, ndarray, ndarray, csr_matrix, int, str, csr_matrix, bool],
    Tuple[ndarray, ndarray, ndarray],
]


def construct_coo_matrix(
    p: ndarray,
    q: ndarray,
    r: Optional[ndarray] = None,
    dtype: str = "float32",
    shape: Optional[Tuple[int, int]] = None,
) -> coo_matrix:
    r = r if r is not None else ones_like(p)
    return coo_matrix((r.astype(dtype), (p, q)), shape=shape)


def construct_dok_matrix(
    p: ndarray,
    q: ndarray,
    r: Optional[ndarray] = None,
    dtype: str = "float32",
    shape: Optional[Tuple[int, int]] = None,
) -> dok_matrix:
    if shape is None:
        shape = (p.max() + 1, q.max() + 1)

    r = r if r is not None else ones_like(p)

    mat = dok_matrix(shape, dtype=dtype)

    for i, k in enumerate(zip(p, q)):
        mat[k] = r[i]

    return mat


def rejection_sample(p: int, q: Union[List, Set]) -> Iterator[int]:
    """
    Sample without replacement from a set of 'p' items that are not in set 'q'.

    This function will continue to draw from 'p' until all elements in 'p' are in 'q'.

    Parameters
    ----------
    p: int
        The number of elements to sample from.
    q: set-like
        An set of items to excluded from sampling, e.g. existing purchases.

    Returns
    -------
    output: Iterator
        A generator yielding integers, where each integer corresponds to a negative
        sample (i.e. a sample from 'p' that is not in 'q').

    """

    sampled = set(q).copy()
    sampled.add(0)

    while len(sampled) < p:
        choice = random.randint(p)
        if choice not in sampled:
            yield choice
            sampled.add(choice)


def relative_rejection_sample(p: int, q: Union[List, Set], k: int,) -> Iterator[int]:
    """
    Sample without replacement from a set of 'p' items that are not in set 'q', where
    the number of samples 'k' is drawn for each element in the set 'q'

    Parameters
    ----------
    p: int
        The number of elements to sample from.
    q: set-like
        An set of items to excluded from sampling, e.g. existing purchases.
    k: int
        The number of samples to draw from 'p' _for each_ element in 'q'.

    Returns
    -------
    output: Iterator
        A generator yielding integers, where each integer corresponds to a negative
        sample (i.e. a sample from 'p' that is not in 'q').

    """

    yield from (a for _ in range(len(q)) for a in islice(rejection_sample(p, q), k))


def single_negative_sample(
    i: int, mat: csr_matrix, k: int, mode: str = "relative"
) -> Iterator[int]:
    """
    Generate single negative sample of at least 'k' elements for user index 'i' in
    interaction matrix 'mat'.

    Parameters
    ----------
    i: int
        The index (in the 'mat') of the 'user' for whom we wish to generate negative
        samples.
    mat: csr_matrix
        A sparse matrix containing interaction mappings (e.g. non-zero elements
        correspond to an user-item interaction, perhaps where a user has purchased or
        rated an item).
    k: int
        The number of samples to draw from items in 'mat'. If mode is 'relative', this
        will sample 'mat' _for each_ element in 'mat' that is non-zero.
    mode: str
        One of two modes:
            'relative' - Sample 'k' negative items from 'p' (total items), _for each_
                         item the user 'i' has interacted with (i.e. for each positive,
                         sample 'k' negatives). This will produce 'k' x 'p' samples.
            'absolute' - Sample 'k' negative items from 'p' (total items). This will
                         produce 'k' samples.

    Returns
    -------
    output: Iterator
        A generator yielding integers, where each integer corresponds to a negative
        sample (i.e. a sample from 'p' that is not in 'q').

    """

    n = mat.shape[1]
    sampled = mat[i].nonzero()[1]

    if mode == "relative":
        yield from relative_rejection_sample(n, sampled, k)
    elif mode == "absolute":
        yield from islice(rejection_sample(n, sampled), k)
    else:
        raise ValueError(f"Unknown negative sampling mode '{mode}'.")


def iter_negative_samples(
    arr: List[int], mat: csr_matrix, k: int, **kwargs: Optional[Any]
) -> Iterator[Tuple[int, int]]:
    """
    Generate single negative samples of at least 'k' elements for each user id 'i' in
    interaction matrix 'mat'.

    Parameters
    ----------
    arr: list
        A list of corresponding ids.
    mat: csr_matrix
        A sparse matrix containing interaction mappings (e.g. non-zero elements
        correspond to an user-item interaction, perhaps where a user has purchased or
        rated an item).
    k: int
        The number of samples to draw from items in 'mat'. If mode is 'relative', this
        will sample 'mat' _for each_ element in 'mat' that is non-zero.
    kwargs: any, optional
        See `single_negative_sample`.

    Returns
    -------
    output: Iterator
        An iterator, where each element corresponds to a negative user-item pair.

    Notes
    -----
    * It's assumed that the row-index of the sparse matrix corresponds to users, while
      the column-index of the matrix corresponds to items, i.e.:
        mat.shape == (n_users, n_items)

    See Also
    --------
    xanthus.dataset.utils.single_negative_sample

    """
    yield from (
        (i, z) for i in arr for z in single_negative_sample(i, mat, k, **kwargs)
    )


def unpack_negative_samples(
    arr: ndarray, mat: csr_matrix, k: int, **kwargs: Optional[Any]
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Generate single negative samples of at least 'k' elements for each user id 'i' in
    interaction matrix 'mat' _as a set of ndarrays_.

    This is a utility function wrapping the behaviour of `iter_negative_samples`.

    Parameters
    ----------
    arr: list
        A list of corresponding ids.
    mat: csr_matrix
        A sparse matrix containing interaction mappings (e.g. non-zero elements
        correspond to an user-item interaction, perhaps where a user has purchased or
        rated an item).
    k: int
        The number of samples to draw from items in 'mat'. If mode is 'relative', this
        will sample 'mat' _for each_ element in 'mat' that is non-zero.
    kwargs: any, optional
        See `single_negative_sample`.

    Returns
    -------
    output: tuple
        A set of three arrays, with the first typically corresponding to user ids, the
        second to item ids, and the third to ratings.

    See Also
    --------
    iter_negative_samples

    """

    arr = list(iter_negative_samples(np.unique(arr), mat, k, **kwargs))
    data = np.asarray(arr).astype(np.int32)
    return data[:, 0], data[:, 1], zeros(shape=data.shape[0])


def sample_negatives(
    users: ndarray,
    items: ndarray,
    ratings: ndarray,
    interactions: csr_matrix,
    negative_samples: int,
    sampling_mode: str = "absolute",
    aux_matrix: Optional[csr_matrix] = None,
    concat: bool = True,
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Draw negative samples from the provided dataset, and optionally return these
    samples concatenated to the provided user/item/rating arrays.

    For the purposes of this function, a negative sample is a sample of an _item_ for
    a _user_ that that user has not interacted with (given the provided parameters).

    Parameters
    ----------
    users: ndarray
        An array containing user IDs.
    items: ndarray
        An array containing item IDs associated with a given user ID.
    ratings: ndarray
        An array corresponding to ratings for a given user-item interaction.
    interactions: csr_matrix
        The sparse matrix storing interactions info.
    negative_samples: int
        The number of negative samples to draw.
    sampling_mode: str
        One of two modes:
            'relative' - Sample 'k' negative items from 'p' (total items), _for each_
                         item the user 'i' has interacted with (i.e. for each positive,
                         sample 'k' negatives). This will produce 'k' x 'p' samples.
            'absolute' - Sample 'k' negative items from 'p' (total items). This will
                         produce 'k' samples.
    aux_matrix: csr_matrix, optional
        Optionally provide a second CSR matrix. This may contain additional interactions
        you wish to utilise in the sampling, but may not be present in the original set.
    concat: bool
        Indicate whether you wish the negative samples to be returned concatenated to
        your input user/item/rating arrays.

    Returns
    -------
    output: tuple
        A set of three arrays, with the first typically corresponding to user ids, the
        second to item ids, and the third to ratings.

    """

    if aux_matrix is not None:
        interactions += aux_matrix

    neg_users, neg_items, neg_ratings = unpack_negative_samples(
        users, interactions, negative_samples, mode=sampling_mode
    )

    if concat:
        users = np.concatenate((users, neg_users))
        items = np.concatenate((items, neg_items))
        ratings = np.concatenate((ratings, neg_ratings))

        return users, items, ratings
    else:
        return neg_users, neg_items, neg_ratings


def groupby(key: ndarray, values: ndarray) -> Tuple[ndarray, List[ndarray]]:
    """
    Execute a simple pure-numpy group-by operation over two arrays.

    Note that both arrays should be of the same type.

    Parameters
    ----------
    key: ndarray
        The array containing keys (values you wish to group on). Assumed to be 1D.
    values: ndarray
        The array containing values (that you wish to group together). Assumed to be 1D.

    Returns
    -------
    output: tuple
        A pair, where the first element is the ordered group keys, and the second a list
        of ndarrays such that each element in order maps to a group key at the same
        index, such that dict(zip(groups, grouped)) would produce a key-value map to the
        groups.

    """

    x = np.c_[key, values]
    idx = np.argsort(x[:, 0])
    x = x[idx]

    groups, counts = np.unique(x[:, 0], return_counts=True)
    grouped = np.split(x[:, 1], np.cumsum(counts)[:-1])

    return groups, grouped


def as_implicit(a: ndarray) -> ndarray:
    """
    Utility method for casting all scores as implicit (i.e. all ones).

    In practice, a sugary alias for `lambda _: ones_like(_)`.

    Parameters
    ----------
    a: ndarray
        An array of floats corresponding to user-item scores.

    Returns
    -------
    output: ndarray
        An array of ones. Simple, really.

    """

    return np.ones_like(a)


def fold(
    df: DataFrame,
    key: str,
    cols: List[str],
    fn: Optional[Callable[[str], Iterator[str]]] = None,
    deduplicate: bool = True,
) -> DataFrame:
    """
    'Fold' a wide DataFrame into a DataFrame in the correct format to be passed to a
    xanthus Dataset (i.e. a DataFrame with two columns, '{key}' and 'tag').

    Parameters
    ----------
    df: DataFrame
        An input dataframe of the format:

            key|cols{0}|...|cols{n}
            :-:|:-----:|:-:|:-----:
            ...

        Where 'key' and 'cols{0}, ..., cols{n}' correspond to the 'key' and 'cols'
        parameter values. See the example below for more details.
    key: str
        The primary key (e.g. unique customer ID) of the input DataFrame.
    cols: list
        A list of column names to be 'folded' (i.e. metadata fields).
    fn: callable, optional
        An optional callable that can be used, for example, to convert a cell in a
        DataFrame into a set of tokens. This can be used on the MovieLens dataset to
        unpack movie genres, for example.
    deduplicate: bool
        Deduplicate the resulting frame to ensure all key-tag pairs occur exactly once.

    Returns
    -------
    output: DataFrame
        A 'folded' DataFrame ready to be used as input to a Dataset

    Examples
    --------

    >>> import pandas as pd
    >>> data = [["jane smith", "london", "doctor"],
    ...         ["dave davey", "manchester", "spaceman"],
    ...         ["john appleseed", "san francisco", "corporate shill"],
    ...         ["jenny boo", "paris", "ninja"]]
    >>> raw_meta = pd.DataFrame(data=data,
    ...                            columns=["user", "location", "occupation"])
    >>> meta = fold(raw_meta, "user", ["location", "occupation"])
    >>> meta
                 user              tag
    0      jane smith           london
    1      jane smith           doctor
    2      dave davey       manchester
    3      dave davey         spaceman
    4  john appleseed    san francisco
    5  john appleseed  corporate shill
    6       jenny boo            paris
    7       jenny boo            ninja

    """

    tags = df[cols].values.tolist()
    keys = df[key].values.tolist()

    if fn is None:
        pairs = ((k, t) for i, k in enumerate(keys) for t in tags[i])
    else:
        pairs = (
            (k, t)
            for i, k in enumerate(keys)
            for t in (e for _ in tags[i] for e in fn(_))
        )

    output = DataFrame(data=pairs, columns=[key, "tag"])

    return output.drop_duplicates([key, "tag"])
