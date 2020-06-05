import pytest
import uuid

import numpy as np
import pandas as pd

from xanthus.dataset import DatasetEncoder, Dataset

np.random.seed(42)


@pytest.fixture
def sample_dataset():
    a = [uuid.uuid4().hex[:10] for _ in range(100)]
    b = [uuid.uuid4().hex[:10] for _ in range(100)]
    return np.c_[np.random.choice(a, 1000), np.random.choice(b, 1000)]


@pytest.fixture
def sample_dataframes(k=1000):
    a = [uuid.uuid4().hex[:10] for _ in range(1000)]
    b = [uuid.uuid4().hex[:10] for _ in range(1000)]
    a_m = [uuid.uuid4().hex[:4] for _ in range(10)]
    b_m = [uuid.uuid4().hex[:4] for _ in range(10)]

    interactions_data = np.c_[
        np.random.choice(a, k), np.random.choice(b, k), np.random.randint(1, 5, size=k),
    ]

    interactions = pd.DataFrame(
        data=interactions_data, columns=["user", "item", "rating"]
    )
    users = pd.DataFrame(
        data=[(_, np.random.choice(a_m)) for _ in np.unique(interactions_data[:, 0])],
        columns=["user", "tag"],
    )
    items = pd.DataFrame(
        data=[(_, np.random.choice(b_m)) for _ in np.unique(interactions_data[:, 1])],
        columns=["item", "tag"],
    )

    return interactions, users, items


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
def test_fit_user_item(sample_dataset):
    d = DatasetEncoder()
    assert d.fit(users=sample_dataset[:, 0],
                 items=sample_dataset[:, 1]) == d


def test_fit_transform_user_item(sample_dataset):
    d = DatasetEncoder()
    d.fit_transform(users=sample_dataset[:, 0],
                    items=sample_dataset[:, 1])


def test_fit_reversible_user_item(sample_dataset):
    d = DatasetEncoder()
    d.fit(sample_dataset)
