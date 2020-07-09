"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import itertools
import functools
import multiprocessing as mp
from typing import Optional, Tuple

import numpy as np

from tensorflow.keras.layers import (
    Layer,
    Input,
    Flatten,
    Embedding,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import RandomNormal, Initializer


def get_embedding_block(
    n_vocab: int,
    n_dim: int,
    n_factors: int,
    reg: float = 0.0,
    initializer: Initializer = RandomNormal(stddev=0.01),
    inputs: Optional[Input] = None,
) -> Tuple[Layer, Embedding, Embedding]:
    """
    Utility function for building Embedding blocks for neural recommenders.

    Parameters
    ----------
    n_vocab: int
        The size of the 'vocabulary' of the Embedding.
    n_dim: int
        The length of the input vectors.
    n_factors: int
        The total number of latent features in the recommender model (output).
    reg: float
        The L2 regularization to apply to the embeddings.
    initializer: Initializer
        The weight initializer for the embeddings.
    inputs: Input, optional
        Optionally provide an 'input' layer for the embedding block.

    Returns
    -------
    output: tuple
        A tuple, where the first element corresponds to the Embeddings' input layer,
        the second a bias term, and the third the embedding layer itself.

    """

    inputs = inputs if inputs is not None else Input(shape=(n_dim,))
    embedding = Embedding(
        input_dim=n_vocab,
        output_dim=n_factors,
        input_length=n_dim,
        embeddings_initializer=initializer,
        embeddings_regularizer=l2(reg),
    )(inputs)

    bias_embedding = Embedding(input_dim=n_vocab, output_dim=1)(inputs)
    factors = Flatten()(embedding)

    return inputs, Flatten()(bias_embedding), factors


def chunked(it, n):
    g = (_ for _ in it)
    c = list(itertools.islice(g, n))
    while c:
        yield c
        c = list(itertools.islice(g, n))


def predict_single(model, ranker, user, items, excluded, count):
    x = [np.ones_like(items) * user, items]
    yh = model(x).flatten()
    return ranker(yh, count, encodings=items, excluded=excluded)


def predict(model, ranker, users, items, excluded, count):
    recommended = []

    for (user, target_items) in zip(users, items):
        yh = predict_single(model, ranker, user, target_items, excluded, count)
        recommended.append(yh)

    return recommended


def rank(w, n, encodings=None, excluded=None):
    ranked = w.argsort()[::-1]

    if encodings is not None:
        ranked = encodings[ranked]

    if excluded is not None:
        ranked = ranked[~np.isin(ranked, excluded)]

    return ranked[:n]


def alt_rank(w, n, excluded=None, encodings=None):
    excluded = excluded if excluded is not None else set()

    count = n + len(excluded)

    if count < len(w):
        ids = np.argpartition(w, -count)[-count:]
        best = sorted(zip(ids, w[ids]), key=lambda x: -x[1])
    else:
        best = sorted(enumerate(w), key=lambda x: -x[1])

    return list(itertools.islice((rec[0] for rec in best if rec[0] not in excluded), n))

