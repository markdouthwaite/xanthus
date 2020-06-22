"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Union, List, Set, Iterator, Tuple, Any
from itertools import islice

from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from numpy import (
    random,
    ones_like,
    ndarray,
    c_,
    split as _split,
    argsort,
    cumsum,
    unique,
)


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


def sample_negatives(
    ids: List[int], mat: csr_matrix, k: int, **kwargs: Optional[Any]
) -> Iterator[Tuple[int, int]]:
    """
    Generate single negative samples of at least 'k' elements for each user id 'i' in
    interaction matrix 'mat'.

    Parameters
    ----------
    ids: list
        A list of corresponding
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
        (i, z) for i in ids for z in single_negative_sample(i, mat, k, **kwargs)
    )


def groupby(a: ndarray, b: ndarray) -> Tuple[ndarray, List[ndarray]]:
    x = c_[a, b]
    ind = argsort(x[:, 0])
    x = x[ind]

    groups, counts = unique(x[:, 0], return_counts=True)
    grouped = _split(x[:, 1], cumsum(counts)[:-1])

    return groups, grouped
