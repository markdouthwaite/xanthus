"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import math
import random

from numpy import isclose

from xanthus.evaluate import score, ndcg


def test_parallel_ndcg_success():
    n = 10000
    predicted = [list(range(10)) for _ in range(n)]
    actual = [list(range(random.randint(10, 30))) for _ in range(n)]

    assert sum(score(ndcg, actual, predicted, n_cpu=8)) == n


def test_ndcg():
    idcg = sum((1.0 / math.log(i+2)) for i in range(4))
    dcg = idcg - (1.0 / math.log(3))
    assert isclose(ndcg([0, 2, 3, 5, 7, 9], [5, 8, 3, 2]), (dcg / idcg))

