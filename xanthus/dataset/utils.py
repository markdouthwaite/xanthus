from typing import Optional, Union, List, Set, Generator, Tuple
from itertools import islice

from scipy.sparse import coo_matrix, csr_matrix, dok_matrix
from numpy import random, ones_like, ndarray


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


def rejection_sample(j: int, sampled: Optional[Union[List, Set]] = None) -> Generator:
    sampled = set(sampled) if sampled is not None else set()

    while True:
        choice = random.randint(j)
        if choice not in sampled:
            yield choice
            sampled.add(choice)


def single_sample_zeros(i: int, mat: csr_matrix, k: int) -> Generator:
    yield from islice(rejection_sample(mat.shape[1], mat[i].nonzero()[1]), k)


def sample_zeros(ids: List[int], mat: csr_matrix, k: int) -> Generator:
    yield from ((i, z) for i in ids for z in single_sample_zeros(i, mat, k))
