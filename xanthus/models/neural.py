"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Any, Tuple
from tensorflow.keras import Model
from tensorflow.keras.layers import Multiply, Dense, Concatenate
from tensorflow.keras.initializers import lecun_uniform
from tensorflow.keras.regularizers import l2

from xanthus.datasets import Dataset
from xanthus.models import utils, base


class MultiLayerPerceptronModel(base.NeuralRecommenderModel):
    """
    An implementation of a Multilayer Perceptron (MLP) model in Keras.

    Parameters
    ----------
    layers: tuple
        A tuple, where each element corresponds to the number of units in each of the
        layers of the MLP.
    activations: str
        The activation function to use for each of the layers in the MLP.
    l2_reg: float
        The L2 regularization to be applied to each of the layers in the MLP.

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569

    See Also
    --------
    xanthus.models.base.NeuralRecommenderModel

    """

    def __init__(
        self,
        *args: Optional[Any],
        layers: Tuple[int, ...] = (64, 32, 16, 8),
        activations: str = "relu",
        l2_reg: float = 1e-3,
        **kwargs: Optional[Any]
    ):
        """Initialize a MultiLayerPerceptronModel."""

        super().__init__(*args, **kwargs)
        self._activations = activations
        self._layers = layers
        self._l2_reg = l2_reg

    def _build_model(
        self,
        dataset: Dataset,
        n_user_dim: int = 1,
        n_item_dim: int = 1,
        n_factors: int = 50,
        **kwargs: Optional[Any]
    ) -> Model:
        """
        Build a Keras model, in this case a MultiLayerPerceptronModel (MLP)
        model. See [1] for more info. The original code released with [1] can be
        found at [2].

        Parameters
        ----------
        dataset: Dataset
            The input dataset. This is used to specify the 'vocab' size of each of the
            'embedding blocks' (of which there are two in this architecture).
        n_user_dim: int
            The dimensionality of the user input vector. When using metadata, you should
            make sure to set this to the size of each of these vectors.
        n_item_dim: int
            The dimensionality of the item input vector. When using metadata, you should
            make sure to set this to the size of each of these vectors.
        n_factors: int
            The dimensionality of the latent feature space _for both users and items_
            for the GMF component of the architecture.

        Returns
        -------
        output: Model
            The 'complete' Keras Model object.

        References
        ----------
        [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569
        [2] https://github.com/hexiangnan/neural_collaborative_filtering
        """

        n_user_vocab = dataset.all_users.shape[0]
        n_item_vocab = dataset.all_items.shape[0]

        if dataset.user_meta is not None:
            n_user_vocab += dataset.user_meta.shape[1]
        if dataset.item_meta is not None:
            n_item_vocab += dataset.item_meta.shape[1]

        # mlp block
        user_input, user_bias, user_factors = utils.get_embedding_block(
            n_user_vocab, n_user_dim, int(self._layers[0] / 2)
        )
        item_input, item_bias, item_factors = utils.get_embedding_block(
            n_item_vocab, n_item_dim, int(self._layers[0] / 2)
        )

        body = Concatenate()([user_factors, item_factors])

        for layer in self._layers:
            body = Dense(
                layer,
                activity_regularizer=l2(self._l2_reg),
                activation=self._activations,
            )(body)

        output = Dense(1, activation="sigmoid", kernel_initializer=lecun_uniform())(
            body
        )

        return Model(inputs=[user_input, item_input], outputs=output)


class NeuralMatrixFactorizationModel(base.NeuralRecommenderModel):
    """
    An implementation of a Neural Matrix Factorization (NeuMF) model in Keras.

    Parameters
    ----------
    layers: tuple
        A tuple, where each element corresponds to the number of units in each of the
        layers of the MLP.
    activations: str
        The activation function to use for each of the layers in the MLP.
    l2_reg: float
        The L2 regularization to be applied to each of the layers in the MLP.

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569

    See Also
    --------
    xanthus.models.base.NeuralRecommenderModel

    """

    def __init__(
        self,
        *args: Optional[Any],
        layers: Tuple[int, ...] = (64, 32, 16, 8),
        activations: str = "relu",
        l2_reg: float = 1e-3,
        **kwargs: Optional[Any]
    ):
        """Initialize a MultiLayerPerceptronModel."""

        super().__init__(*args, **kwargs)
        self._activations = activations
        self._layers = layers
        self._l2_reg = l2_reg

    def _build_model(
        self,
        dataset: Dataset,
        n_user_dim: int = 1,
        n_item_dim: int = 1,
        n_factors: int = 50,
        **kwargs: Optional[Any]
    ) -> Model:
        """
        Build a Keras model, in this case a NeuralMatrixFactorizationModel (NeuMF)
        model. This is a recommender model with two input branches (one half the same
        architecture as in GeneralizedMatrixFactorizationModel, the other the same
        architecture as in MultiLayerPerceptronModel. See [1] for more info. The
        original code released with [1] can be found at [2].

        Parameters
        ----------
        dataset: Dataset
            The input dataset. This is used to specify the 'vocab' size of each of the
            'embedding blocks' (of which there are four in this architecture).
        n_user_dim: int
            The dimensionality of the user input vector. When using metadata, you should
            make sure to set this to the size of each of these vectors.
        n_item_dim: int
            The dimensionality of the item input vector. When using metadata, you should
            make sure to set this to the size of each of these vectors.
        n_factors: int
            The dimensionality of the latent feature space _for both users and items_
            for the GMF component of the architecture.

        Returns
        -------
        output: Model
            The 'complete' Keras Model object.

        References
        ----------
        [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569
        [2] https://github.com/hexiangnan/neural_collaborative_filtering

        """

        n_user_vocab = dataset.all_users.shape[0]
        n_item_vocab = dataset.all_items.shape[0]

        if dataset.user_meta is not None:
            n_user_vocab += dataset.user_meta.shape[1]
        if dataset.item_meta is not None:
            n_item_vocab += dataset.item_meta.shape[1]

        # mlp block
        user_input, mlp_user_bias, mlp_user_factors = utils.get_embedding_block(
            n_user_vocab, n_user_dim, int(self._layers[0] / 2)
        )
        item_input, mlp_item_bias, mlp_item_factors = utils.get_embedding_block(
            n_item_vocab, n_item_dim, int(self._layers[0] / 2)
        )

        mlp_body = Concatenate()([mlp_user_factors, mlp_item_factors])

        for layer in self._layers:
            mlp_body = Dense(
                layer,
                activity_regularizer=l2(self._l2_reg),
                activation=self._activations,
            )(mlp_body)

        # mf block
        user_input, mf_user_bias, mf_user_factors = utils.get_embedding_block(
            n_user_vocab, n_user_dim, n_factors, inputs=user_input,
        )
        item_input, mf_item_bias, mf_item_factors = utils.get_embedding_block(
            n_item_vocab, n_item_dim, n_factors, inputs=item_input,
        )
        mf_body = Multiply()([mf_user_factors, mf_item_factors])

        body = Concatenate()([mf_body, mlp_body])

        output = Dense(1, activation="sigmoid", kernel_initializer=lecun_uniform())(
            body
        )

        return Model(inputs=[user_input, item_input], outputs=output)


class GeneralizedMatrixFactorizationModel(base.NeuralRecommenderModel):
    """
    An implementation of a Generalized Matrix Factorization (GMF) model in Keras.

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569

    """

    def _build_model(
        self,
        dataset: Dataset,
        n_user_dim: int = 1,
        n_item_dim: int = 1,
        n_factors: int = 50,
        **kwargs: Optional[Any]
    ) -> Model:
        """
        Build a Keras model, in this case a GeneralizedMatrixFactorizationModel (GMF)
        model. See [1] for more info. The original code released with [1] can be
        found at [2].

        Parameters
        ----------
        dataset: Dataset
            The input dataset. This is used to specify the 'vocab' size of each of the
            'embedding blocks' (of which there are two in this architecture).
        n_user_dim: int
            The dimensionality of the user input vector. When using metadata, you should
            make sure to set this to the size of each of these vectors.
        n_item_dim: int
            The dimensionality of the item input vector. When using metadata, you should
            make sure to set this to the size of each of these vectors.
        n_factors: int
            The dimensionality of the latent feature space _for both users and items_
            for the GMF component of the architecture.

        Returns
        -------
        output: Model
            The 'complete' Keras Model object.

        References
        ----------
        [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569
        [2] https://github.com/hexiangnan/neural_collaborative_filtering
        """

        n_user_vocab = dataset.all_users.shape[0]
        n_item_vocab = dataset.all_items.shape[0]

        if dataset.user_meta is not None:
            n_user_vocab += dataset.user_meta.shape[1]
        if dataset.item_meta is not None:
            n_item_vocab += dataset.item_meta.shape[1]

        user_input, user_bias, user_factors = utils.get_embedding_block(
            n_user_vocab, n_user_dim, n_factors, **kwargs
        )
        item_input, item_bias, item_factors = utils.get_embedding_block(
            n_item_vocab, n_item_dim, n_factors, **kwargs
        )

        body = Multiply()([user_factors, item_factors])
        output = Dense(1, activation="sigmoid", kernel_initializer=lecun_uniform())(
            body
        )

        return Model(inputs=[user_input, item_input], outputs=output)
