"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""


from typing import Optional, Any, NoReturn, Iterable, List, Union, Callable, Tuple

from tensorflow import Tensor
from tensorflow.keras import Model
from tensorflow.keras.layers import Multiply, Dense, Concatenate, Layer
from tensorflow.keras.initializers import lecun_uniform
from tensorflow.keras.regularizers import Regularizer
from .utils import InputEmbeddingBlock

Activation = Callable[[Tensor], Tensor]


class GeneralizedMatrixFactorization(Model):
    """
    A Keras model implementing Generalized Matrix Factorization (GMF) architecture
    from [1].

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569
    """

    def __init__(
        self,
        n: int,
        m: int,
        factors: int = 32,
        *args: Any,
        embedding_regularizer: Optional[Regularizer] = None,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Initialize a GMF model.

        Parameters
        ----------
        n: int
            The size of the 'vocabulary' of users (i.e. unique user IDs + metadata tags)
        m: int
            The size of the 'vocabulary' of items (i.e. unique item IDs + metadata tags)
        factors: int
            The size of 'predictive factors' (analogous to latent features) of the MF
            model.
        embedding_regularizer: Regularizer
            A regularizer to be applied to Embdeddings. See keras.layers.Embedding
        args: Any, optional
            Optional args to be passed to base keras.Model.
        kwargs: Any
            Optional kwargs to be passed to base keras.Model

        """

        super().__init__(*args, **kwargs)
        self.user_embedding = InputEmbeddingBlock(
            n, factors, name="user_embeddings", regularizer=embedding_regularizer
        )
        self.item_embedding = InputEmbeddingBlock(
            m, factors, name="item_embeddings", regularizer=embedding_regularizer
        )
        self.multiply = Multiply(name="multiply")
        self.prediction = Dense(
            1,
            activation="sigmoid",
            kernel_initializer=lecun_uniform(),
            name="prediction",
        )

    def call(
        self,
        inputs: Union[List[Tensor], Tensor],
        training: Optional[bool] = None,
        mask: Optional[Union[List[Tensor], Tensor]] = None,
    ) -> Union[List[Tensor], Tensor]:
        """Call the model."""

        user_z = self.user_embedding(inputs[0])
        item_z = self.item_embedding(inputs[1])
        z = self.multiply([user_z, item_z])
        return self.prediction(z)


class MultiLayerPerceptron(Model):
    """
    A Keras model implementing Multilayer Perceptron Model (MLP) recommendation model
    architecture described in [1].

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569
    """

    def __init__(
        self,
        n: int,
        m: int,
        layers: Tuple[int] = (32, 16, 8),
        regularizer: Optional[Regularizer] = None,
        embedding_regularizer: Optional[Regularizer] = None,
        activation: Union[str, Activation] = "relu",
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Initialize a GMF model.

        Parameters
        ----------
        n: int
            The size of the 'vocabulary' of users (i.e. unique user IDs + metadata tags)
        m: int
            The size of the 'vocabulary' of items (i.e. unique item IDs + metadata tags)
        layers: tuple
            A tuple, where each element corresponds to the number of units in each of
            the layers of the MLP.
        regularizer: Regularizer
            A regularizer to be applied to hidden layers. See keras.layers.Dense
        embedding_regularizer: Regularizer
            A regularizer to be applied to Embdeddings. See keras.layers.Embedding
        activation: str, Regularizer
            The activation function to use for hidden layers.
        args: Any, optional
            Optional args to be passed to base keras.Model.
        kwargs: Any
            Optional kwargs to be passed to base keras.Model

        """

        super().__init__(*args, **kwargs)

        self.user_embedding = InputEmbeddingBlock(
            n, layers[0] // 2, name="user_embeddings", regularizer=embedding_regularizer
        )
        self.item_embedding = InputEmbeddingBlock(
            m, layers[0] // 2, name="item_embeddings", regularizer=embedding_regularizer
        )
        self.concat = Concatenate(name="concat")
        self.hidden = list(self._build_layers(layers, activation, regularizer))
        self.prediction = Dense(
            1,
            activation="sigmoid",
            kernel_initializer=lecun_uniform(),
            name="prediction",
        )

    @staticmethod
    def _build_layers(
        layers: List[int], activation: Union[str, Activation], regularizer: Regularizer
    ) -> Iterable[Layer]:
        """Build the model's hidden layers."""

        for i, layer in enumerate(layers):
            yield Dense(
                layer,
                activity_regularizer=regularizer,
                activation=activation,
                name=f"layer{i+1}",
            )

    def call(
        self,
        inputs: Union[List[Tensor], Tensor],
        training: Optional[bool] = None,
        mask: Optional[Union[List[Tensor], Tensor]] = None,
    ) -> Union[List[Tensor], Tensor]:
        """Invoke the model. A single 'forward pass'."""

        user_z = self.user_embedding(inputs[0])
        item_z = self.item_embedding(inputs[1])
        z = self.concat([user_z, item_z])

        for layer in self.hidden:
            z = layer(z)

        return self.prediction(z)


class NeuralMatrixFactorization(Model):
    """
    A Keras model implementing Neural Matrix Factorization (GMF) architecture
    from [1].

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569
    """

    def __init__(
        self,
        n: int,
        m: int,
        factors: int = 32,
        layers: Tuple[int] = (32, 16, 8),
        activation: Union[str, Activation] = "relu",
        regularizer: Optional[Regularizer] = None,
        embedding_regularizer: Optional[Regularizer] = None,
        *args: Any,
        **kwargs: Any,
    ) -> NoReturn:
        """
        Initialize a NMF model.

        Parameters
        ----------
        n: int
            The size of the 'vocabulary' of users (i.e. unique user IDs + metadata tags)
        m: int
            The size of the 'vocabulary' of items (i.e. unique item IDs + metadata tags)
        layers: tuple
            A tuple, where each element corresponds to the number of units in each of
            the layers of the MLP.
        regularizer: Regularizer
            A regularizer to be applied to hidden layers. See keras.layers.Dense
        embedding_regularizer: Regularizer
            A regularizer to be applied to Embdeddings. See keras.layers.Embedding
        activation: str, Regularizer
            The activation function to use for hidden layers.
        args: Any, optional
            Optional args to be passed to base keras.Model.
        kwargs: Any
            Optional kwargs to be passed to base keras.Model

        """

        super().__init__(*args, **kwargs)
        self.gmf_user_embedding = InputEmbeddingBlock(
            n, factors, regularizer=embedding_regularizer
        )
        self.gmf_item_embedding = InputEmbeddingBlock(
            m, factors, regularizer=embedding_regularizer
        )
        self.gmf_multiply = Multiply()

        # check units -- this could cause issues.
        self.mlp_user_embedding = InputEmbeddingBlock(
            n, layers[0] // 2, regularizer=embedding_regularizer
        )
        self.mlp_item_embedding = InputEmbeddingBlock(
            m, layers[0] // 2, regularizer=embedding_regularizer
        )
        self.mlp_concat = Concatenate()
        self.mlp_hidden = list(self._build_layers(layers, activation, regularizer))

        self.concat = Concatenate()
        self.prediction = Dense(
            1,
            activation="sigmoid",
            kernel_initializer=lecun_uniform(),
            name="prediction",
        )

    @staticmethod
    def _build_layers(
        layers: Iterable[int],
        activation: Union[str, Activation],
        regularizer: Regularizer,
    ) -> Iterable[Dense]:
        """Build the model's hidden layers."""

        for i, layer in enumerate(layers):
            yield Dense(
                layer,
                activity_regularizer=regularizer,
                activation=activation,
                name=f"layer{i+1}",
            )

    def call(
        self,
        inputs: Union[List[Tensor], Tensor],
        training: Optional[bool] = None,
        mask: Optional[Union[List[Tensor], Tensor]] = None,
    ) -> Union[List[Tensor], Tensor]:
        """Invoke the model. A single 'forward pass'."""

        mlp_user_z = self.mlp_user_embedding(inputs[0])
        mlp_item_z = self.mlp_item_embedding(inputs[1])
        mlp_z = self.mlp_concat([mlp_user_z, mlp_item_z])

        for layer in self.mlp_hidden:
            mlp_z = layer(mlp_z)

        gmf_user_z = self.gmf_user_embedding(inputs[0])
        gmf_item_z = self.gmf_item_embedding(inputs[1])
        gmf_z = self.gmf_multiply([gmf_user_z, gmf_item_z])

        z = self.concat([gmf_z, mlp_z])

        return self.prediction(z)
