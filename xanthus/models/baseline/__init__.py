"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from functools import partial

from .pop_rank import PopRankModel
from .mf import MatrixFactorizationModel

AlternatingLeastSquaresModel = partial(MatrixFactorizationModel, method="als")
BayesianPersonalizedRankingModel = partial(MatrixFactorizationModel, method="bpr")
