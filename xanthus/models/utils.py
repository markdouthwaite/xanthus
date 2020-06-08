"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import (
    Dense,
    Input,
    Flatten,
    Embedding,
    Add,
    Dot,
    Concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam


def get_embedding_block(
    n_vocab, n_dim, n_factors, reg=0.0, initializer="he_normal", inputs=None
):
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


def get_nmf_model(
    n_users,
    n_items,
    n_user_dim=1,
    n_item_dim=1,
    layers=(32, 16, 8),
    n_factors=50,
    l2_reg=0.0,
):
    user_input, mlp_user_bias, mlp_user_factors = get_embedding_block(
        n_users, n_user_dim, n_factors
    )
    item_input, mlp_item_bias, mlp_item_factors = get_embedding_block(
        n_items, n_item_dim, n_factors
    )

    mlp_body = Concatenate()(
        [mlp_user_factors, mlp_user_bias, mlp_item_factors, mlp_item_bias]
    )

    for layer in layers:
        mlp_body = Dense(layer, activity_regularizer=l2(l2_reg), activation="relu")(
            mlp_body
        )

    user_input, mf_user_bias, mf_user_factors = get_embedding_block(
        n_users, n_user_dim, n_factors, inputs=user_input,
    )
    item_input, mf_item_bias, mf_item_factors = get_embedding_block(
        n_items, n_item_dim, n_factors, inputs=item_input,
    )
    mf_body = Dot(-1)([mf_user_factors, mf_item_factors])
    mf_body = Add()([mf_body, mf_user_bias, mf_item_bias])

    body = Concatenate()([mf_body, mlp_body])

    output = Dense(1, activation="sigmoid",)(body)

    return Model(inputs=[user_input, item_input], outputs=output)


def get_mlp_model(
    n_users,
    n_items,
    n_user_dim=1,
    n_item_dim=1,
    layers=(32, 16, 8),
    n_factors=50,
    l2_reg=0.0,
):
    user_input, user_bias, user_factors = get_embedding_block(
        n_users, n_user_dim, n_factors
    )
    item_input, item_bias, item_factors = get_embedding_block(
        n_items, n_item_dim, n_factors
    )

    body = Concatenate()([user_factors, user_bias, item_factors, item_bias])

    for layer in layers:
        body = Dense(layer, activity_regularizer=l2(l2_reg), activation="relu")(body)

    output = Dense(1, activation="sigmoid")(body)

    return Model(inputs=[user_input, item_input], outputs=output)


def get_mf_model(n_users, n_items, n_user_dim=1, n_item_dim=1, n_factors=50):
    user_input, user_bias, user_factors = get_embedding_block(
        n_users, n_user_dim, n_factors
    )
    item_input, item_bias, item_factors = get_embedding_block(
        n_items, n_item_dim, n_factors
    )
    body = Dot(-1)([user_factors, item_factors])
    body = Add()([body, user_bias, item_bias])
    output = Dense(1, activation="sigmoid")(body)

    return Model(inputs=[user_input, item_input], outputs=output)
