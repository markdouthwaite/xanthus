"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import pytest

import numpy as np
import pandas as pd

from xanthus.dataset import DatasetEncoder, Dataset


def test_user_item_end_to_end_mapping(sample_dataframes):
    df, _, _ = sample_dataframes

    df = df.drop_duplicates(subset=["user", "item"]).sort_values(by="user")

    dataset = Dataset.from_frame(df)

    users, items, ratings = dataset.to_arrays()

    decoded = dataset.encoder.inverse_transform(
        users=users.flatten(), items=items.flatten()
    )

    users, items = decoded["users"], decoded["items"]

    assert users.shape[0] == df.shape[0]
    assert items.shape[0] == df.shape[0]

    new = pd.DataFrame(data=np.c_[users, items], columns=["user", "item"])
    new["rating"] = ratings

    for user, group in new.groupby("user"):
        old_group = df[df["user"] == user].sort_values(by="item")
        group = group.sort_values(by="item")
        items = group["item"].values
        old_items = old_group["item"].values
        new_ratings = np.log((1.0 / group["rating"].values) + 1.0)
        old_ratings = old_group["rating"].values.astype(float)
        assert group.shape == old_group.shape
        assert np.all(old_items == items)
        assert np.allclose(new_ratings, old_ratings)


def test_consistent_interactions_size_for_split_datasets(sample_dataframes):
    interactions, _, _ = sample_dataframes

    encoder = DatasetEncoder()
    encoder.fit(interactions["user"], interactions["item"])

    train_interactions = interactions.iloc[: int(len(interactions) / 2)]
    test_interactions = interactions.iloc[: int(len(interactions) / 2)]

    train_dataset = Dataset.from_frame(train_interactions, encoder=encoder)
    test_dataset = Dataset.from_frame(test_interactions, encoder=encoder)
    dataset = Dataset.from_frame(interactions)
    assert dataset.interactions.shape == train_dataset.interactions.shape
    assert dataset.interactions.shape == test_dataset.interactions.shape


@pytest.mark.usefixtures("sample_dataframes")
def test_user_meta_item_end_to_end_mapping(sample_dataframes):
    df, user, item = sample_dataframes

    dataset = Dataset.from_frame(df, user, item)
    users, items, ratings = dataset.to_arrays(output_dim=4)

    print(users)

    # decoded = dataset.encoder.inverse_transform(
    #     users=users.flatten(), items=items.flatten()
    # )
    #
    # users, items = decoded["users"], decoded["items"]


# encoder
