"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import pytest

import numpy as np
import pandas as pd

from xanthus.datasets import DatasetEncoder, Dataset


"""
Test with one set of item metadata
Test with one set of user metadata
Test with both user and item metadata
"""


def dataset_inverse_transform(df):
    df = df.drop_duplicates(subset=["user", "item"]).sort_values(by="user")

    dataset = Dataset.from_df(df)

    users, items, ratings = dataset.to_components()

    decoded = dataset.encoder.inverse_transform(
        users=users.flatten(), items=items.flatten()
    )

    users, items = decoded["users"], decoded["items"]
    return users, items, ratings, df


def test_to_arrays_inverse_transform_shape(sample_dataframes):
    """
    Dataset.to_arrays preserves the original shape of the input frame
    (i.e. has the same number of user-item records).
    """

    df, _, _ = sample_dataframes

    users, items, ratings, df = dataset_inverse_transform(df)

    assert users.shape[0] == df.shape[0]
    assert items.shape[0] == df.shape[0]


def test_to_arrays_inverse_transform_mapping(sample_dataframes):
    """
    Dataset.to_arrays preserves the information in the input frame
    (i.e. the user-item mappings remain the same)

    Notes
    -----
    * User-item mappings may not remain in the same order, so we don't test for that.

    """

    df, _, _ = sample_dataframes

    users, items, ratings, df = dataset_inverse_transform(df)

    new = pd.DataFrame(data=np.c_[users, items], columns=["user", "item"])
    new["rating"] = ratings

    for user, group in new.groupby("user"):
        old_group = df[df["user"] == user].sort_values(by="item")
        group = group.sort_values(by="item")
        items = group["item"].values
        old_items = old_group["item"].values

        assert group.shape == old_group.shape
        assert np.all(old_items == items)
        # assert np.allclose(new_ratings, old_ratings)


def test_shared_encoder_interactions_shapes(sample_dataframes):
    """
    Datasets using the same DatasetEncoder produce interactions matrices of the
    same shape.
    """
    interactions, _, _ = sample_dataframes

    encoder = DatasetEncoder()
    encoder.fit(interactions["user"], interactions["item"])

    train_interactions = interactions.iloc[: int(len(interactions) / 2)]
    test_interactions = interactions.iloc[: int(len(interactions) / 2)]

    train_dataset = Dataset.from_df(train_interactions, encoder=encoder)
    test_dataset = Dataset.from_df(test_interactions, encoder=encoder)
    dataset = Dataset.from_df(interactions)

    assert dataset.interactions.shape == train_dataset.interactions.shape
    assert dataset.interactions.shape == test_dataset.interactions.shape


def test_to_arrays_absolute_negative_sample_shapes(sample_dataframes):
    """
    Dataset.to_arrays produces the correct number of negative samples.
    """

    df, _, _ = sample_dataframes
    dataset = Dataset.from_df(df)

    n_users = dataset.users.shape[0]

    users, items, _ = dataset.to_components()

    for i in range(1, 5):
        sampled_users, sampled_items, _ = dataset.to_components(
            negative_samples=i, sampling_mode="absolute"
        )
        assert sampled_users.shape[0] == (i * n_users) + users.shape[0]
        assert sampled_items.shape[0] == (i * n_users) + users.shape[0]


def test_to_arrays_relative_negative_sample_shapes(sample_dataframes):
    """
    Dataset.to_arrays produces the correct number of negative samples.
    """
    #
    # df, _, _ = sample_dataframes
    # dataset = Dataset.from_df(df)
    #
    # n_users = dataset.users.shape[0]
    #
    # users, items, _ = dataset.to_arrays()
    #
    # for i in range(1, 5):
    #     sampled_users, sampled_items, _ = dataset.to_arrays(negative_samples=i,
    #                                                         sampling_mode="absolute")
    #     assert sampled_users.shape[0] == (i * n_users) + users.shape[0]
    #     assert sampled_items.shape[0] == (i * n_users) + users.shape[0]