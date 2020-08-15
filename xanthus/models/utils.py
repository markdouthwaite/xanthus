"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Tuple, Any, Union, Dict, List

import numpy as np
from numpy import ndarray

from tensorflow.keras.layers import (
    Layer,
    Input,
    Flatten,
    Embedding,
)
from tensorflow.keras.regularizers import l2, Regularizer
from tensorflow.keras.initializers import RandomNormal, Initializer


class InputEmbeddingBlock(Layer):
    """An input embedding block that flattens the output."""

    def __init__(
        self,
        n_vocab: int,
        n_factors: int,
        *args: Any,
        regularizer: Optional[Regularizer] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the block!"""

        super().__init__(*args, **kwargs)
        self._n_factors = n_factors
        self._n_vocab = n_vocab
        self._embedding: Optional[Embedding] = None
        self._output: Optional[Flatten] = None
        self._regularizer: Optional[Union[Regularizer, str]] = regularizer

    def build(self, input_shape: Tuple[int, ...]) -> None:
        """Build the block!"""

        self._embedding = Embedding(
            input_dim=self._n_vocab,
            output_dim=self._n_factors,
            input_length=input_shape,
            embeddings_initializer=RandomNormal(stddev=0.01),
            embeddings_regularizer=self._regularizer,
        )
        self._output = Flatten()

    def call(self, inputs: ndarray, **kwargs: Any) -> ndarray:
        """Call the block."""

        if self._embedding is None or self._output is None:
            raise ValueError(
                "You must call 'build' on an InputEmbeddingBlock before 'call'."
            )
        else:
            x = self._embedding(inputs)
            return self._output(x)


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


def reshape_recommended(
    users: ndarray, items: ndarray, scores: ndarray, n: int, mode: str = "array"
) -> Union[ndarray, Dict[int, List[Tuple[int, float]]]]:
    """
    Reshape recommendations from 'long' format into 'wide' format.
    """

    recommended: Dict[int, List[Tuple[int, float]]] = {k: [] for k in users[:, 0]}

    for user, item, rating in zip(users[:, 0], items[:, 0], scores.flatten()):
        recommended[user].append((item, rating))

    if mode == "dict":
        return recommended

    elif mode == "array":
        return np.asarray(
            [
                [e for e, _ in sorted(recommended[_], key=lambda x: -x[1])][:n]
                for _ in recommended.keys()
            ]
        )
    else:
        raise ValueError(f"Unknown create recommended mode '{mode}'.")
