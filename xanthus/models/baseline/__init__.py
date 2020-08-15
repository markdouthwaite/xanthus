"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from functools import partial

from .pop_rank import PopRank
from .mf import MatrixFactorization

AlternatingLeastSquares = partial(MatrixFactorization, method="als")
BayesianPersonalizedRanking = partial(MatrixFactorization, method="bpr")

__all__ = [
    "MatrixFactorization",
    "AlternatingLeastSquares",
    "BayesianPersonalizedRanking",
    "PopRank",
]
