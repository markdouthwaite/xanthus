import uuid

import pytest
import numpy as np
import pandas as pd


np.random.seed(1)


@pytest.fixture
def sample_dataset():
    a = [uuid.uuid4().hex[:10] for _ in range(100)]
    b = [uuid.uuid4().hex[:10] for _ in range(100)]
    return np.c_[np.random.choice(a, 1000), np.random.choice(b, 1000)]


@pytest.fixture
def sample_dataframes(k=10000):
    a = [uuid.uuid4().hex[:10] for _ in range(1000)]
    b = [uuid.uuid4().hex[:10] for _ in range(500)]
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
