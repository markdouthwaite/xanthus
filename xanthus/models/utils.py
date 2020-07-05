"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Tuple

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
