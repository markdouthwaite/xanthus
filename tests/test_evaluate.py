"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import math
import random

import pytest

from numpy import isclose, unique, all, intersect1d, ones_like
from pandas import DataFrame

from xanthus.datasets import DatasetEncoder, Dataset, groupby
from xanthus.evaluate import score, ndcg, utils, he_sampling, split


@pytest.fixture
def small_split_dataset():
    data = [
        ["0", "0"],
        ["0", "1"],
        ["0", "2"],
        ["1", "0"],
        ["1", "3"],
        ["2", "1"],
        ["2", "4"],
        ["2", "5"],
        ["2", "6"],
        ["3", "1"],
        ["3", "2"],
        ["4", "0"],
        ["5", "0"],
        ["5", "2"],
        ["5", "6"],
        ["6", "7"],
        ["6", "8"],
        ["6", "9"],
        ["7", "2"],
        ["7", "9"],
    ]
    return DataFrame(data=data, columns=["user", "item"])


def test_parallel_ndcg_success():
    n = 10000
    predicted = [list(range(10)) for _ in range(n)]
    actual = [list(range(random.randint(10, 30))) for _ in range(n)]

    assert sum(score(ndcg, actual, predicted, n_cpu=8)) == n


def test_ndcg_example():
    idcg = sum((1.0 / math.log(i + 2)) for i in range(4))
    dcg = idcg - (1.0 / math.log(3))
    assert isclose(ndcg([0, 2, 3, 5, 7, 9], [5, 8, 3, 2], k=6), (dcg / idcg))


def test_split_output_shape(small_split_dataset):
    train, test = utils.split(small_split_dataset, frac_train=0.75)

    assert len(train) + len(test) == len(small_split_dataset)
    assert train.shape[1] == small_split_dataset.shape[1]
    assert test.shape[1] == small_split_dataset.shape[1]


def test_split_output_correctness(small_split_dataset):
    train, test = utils.split(small_split_dataset, frac_train=0.75)

    # check appear correct no. of times.
    block = train.append(test)
    users, counts = unique(block["user"].values, return_counts=True)
    origin_users, origin_counts = unique(
        small_split_dataset["user"].values, return_counts=True
    )

    assert all(origin_counts == counts)
    assert all(origin_users == users)

    # check users in test appear more than once in the original dataset
    assert (
        small_split_dataset["user"].isin(test["user"].unique()).sum() >= len(test) * 2
    )

    # check each hold-out sample is original dataset for that user.
    for user, group in small_split_dataset.groupby("user"):
        other = test[test["user"] == user]
        if other.shape[0] > 0:
            assert len(other.merge(group)) == len(other)


def test_he_sampling_correctness(sample_dataframes):
    df, _, _ = sample_dataframes

    df = df.drop_duplicates()

    encoder = DatasetEncoder()
    encoder.fit(df["user"], df["item"])

    train, test = split(df, n_test=1)

    dataset = Dataset.from_df(df, encoder=encoder, normalize=lambda _: ones_like(_))

    train_dataset = Dataset.from_df(
        train, encoder=encoder, normalize=lambda _: ones_like(_)
    )
    test_dataset = Dataset.from_df(
        test, encoder=encoder, normalize=lambda _: ones_like(_)
    )

    users, items = he_sampling(test_dataset, train_dataset)

    a, b, _ = dataset.to_components()
    all_users, all_items = groupby(a, b)

    for i in range(len(items)):
        # items includes no more than one element from 'all_items'.
        assert len(intersect1d(items[i], all_items[i])) == 1


def test_split_min_records(small_split_dataset):
    train, test = utils.split(small_split_dataset, frac_train=0.75, min_records=2)

    # user '4' appears once, check it doesn't exist
    assert (train["user"] == "4").sum() == 0


def test_split_ignore_users(small_split_dataset):
    # todo: unstable test - should pass for any seed (it doesn't currently).

    frac = 0.5
    train, test = utils.split(
        small_split_dataset, frac_train=0.75, frac_ignored_users=frac
    )

    assert len(train) + len(test) <= int(frac * len(small_split_dataset))
