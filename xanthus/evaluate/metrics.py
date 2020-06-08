"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import math
import multiprocessing as mp
from functools import partial

from typing import List, Callable, Tuple, Any, Optional

from numpy import asarray, isin, unique


def _parfn(fn: Callable[[List[int], List[int]], float], args: Tuple[Any]) -> float:
    """
    A utility function for unpacking arguments when a function is called inside a
    parallel map.
    """
    return fn(*args)


def score(
    fn: Callable[[List[int], List[int], Optional[Any]], float],
    actual: List[List[int]],
    predicted: List[List[int]],
    n_cpu: int = -1,
    **kwargs: Optional[Any],
) -> List[float]:
    """
    Score a set of predicted documents (e.g. recommendations) against 'actual' relevant
    documents (e.g. held-out transaction history).

    This is currently compatible with:
    * precision_at_k
    * normalized_discounted_cumulative_gain

    For 'coverage_at_k', use that function directly.

    Parameters
    ----------
    fn: callable
        A scoring function. This should take at least two arguments: an array-like
        object containing up to 'M' relevant documents, and a second array-like object
        containing 'K' predicted relevant documents.
    actual: array-like
        An array-like object where each element contains up to 'M' elements. Not that
        this means this object may contain elements that are different lengths. This
        is expected. These are collections of relevant documents.
    predicted: array-like
        An array-like object where each element contains 'K' elements. These are each
        collections of documents that are expected to be relevant.
    n_cpu: int
        Indicate whether to utilise multiple cores. By default, n_cpu < 2 indicates the
        execution should proceed within a single process.
    kwargs: optional
        Additional keyword arguments to be passed to the provided metric 'fn'.

    Returns
    -------
    output: list
        A list of floats, where each float corresponds to the computed metric for each
        actual-predicted pair.

    """

    if len(actual) != len(predicted):
        raise ValueError(
            "Total number of results does not match the total number of "
            "expected outputs."
        )

    if n_cpu <= 2:
        return [fn(a, p, **kwargs) for a, p in zip(actual, predicted)]
    else:
        pool = mp.Pool(processes=n_cpu)
        results = pool.map(partial(_parfn, fn, **kwargs), zip(actual, predicted))
        return results


def coverage_at_k(actual: List[int], predicted: List[List[int]], k) -> float:
    """
    Compute the coverage at k (c@k) for a given set of documents (actual) given all
    predictions.

    This metric is useful for checking how many items from across a catalog or
    returned in the top 'k' results.

    Parameters
    ----------
    actual: list, array-like
        An array of containing relevant documents (e.g. 'actual' user activity).
    predicted: list, array-like
        An array of containing lists of predicted documents (e.g. recommendations).
    k: int
        The total number of predicted results to consider.

    Returns
    -------
    output: float
        The c@k for the given element sets.

    """

    n = isin(unique(asarray(predicted)[:, :k]), actual).sum()
    return n / len(actual)


def precision_at_k(actual: List[int], predicted: List[int], k: int = 6) -> float:
    """
    Compute the precision at k (p@k) for a given set of documents.

    For example, precision at 5 corresponds to the number of relevant results among the
    top 5 expected results.

    Parameters
    ----------
    actual: list, array-like
        An array of containing relevant documents (e.g. 'actual' user activity).
    predicted: list, array-like
        An array of containing predicted documents (e.g. recommendations).
    k: int
        The total number of predicted results to consider.

    Returns
    -------
    output: float
        The p@k for the given element sets.

    References
    ----------
        [1] https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/
        [2] https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    """

    hits: float = 0.0

    for i, _ in enumerate(predicted[:k]):

        if _ in actual:
            hits += 1.0

    return hits / min(len(actual), k)


def normalized_discounted_cumulative_gain(
    actual: List[int], predicted: List[int],
) -> float:
    """
    Compute the nDCG for a set of documents assuming binary relevance scores (i.e.
    the presence of an 'actual' element in the predicted elements has relevance '1').

    Examples
    --------

    Consider a set of documents given by the indices:

    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

    And that the following six documents are relevant to a search query (actual):

    {0, 2, 3, 5, 7, 9}

    And four query responses drawn from these documents (predicted):

    {5, 8, 3, 2}

    Then the relevance of the 4 response documents would be:

    {1, 0, 1, 1}

    The DCG would then be: SUM((1 / log(2)) + (1 / log(4)) + (1 / log(5)))
    And the IDCG: SUM((1 / log(2)) + (1 / log(3)) + (1 / log(4)) + (1 / log(5)))

    Where DCG is the Discounted Cumulative Gain, and IDCG is the Ideal DCG (best-
    possible DCG).

    Then we have the normalized metric: nDCG = DCG / IDCG

    Parameters
    ----------
    actual: list, array-like
        An array of containing relevant documents (e.g. 'actual' user activity).
    predicted: list, array-like
        An array of containing predicted documents (e.g. recommendations).

    Returns
    -------
    output: float
        The nDCG for the given element sets.

    References
    ----------
        [1] https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    """

    dcg: float = 0.0
    idcg: float = 0.0
    for i, _ in enumerate(predicted):
        dcg_i = 1 / math.log(i + 2)

        if _ in actual:
            dcg += dcg_i

        idcg += dcg_i

    return dcg / idcg


cak = coverage_at_k
pak = precision_at_k
ndcg = normalized_discounted_cumulative_gain