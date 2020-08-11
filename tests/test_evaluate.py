"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import math
import random

import pytest

from numpy import (
    isclose,
    unique,
    all,
    intersect1d,
    ones_like,
    c_,
    arange,
    asarray,
    ones,
    isin,
)
from pandas import DataFrame

from xanthus.datasets import DatasetEncoder, Dataset, groupby, fold
from xanthus.evaluate import (
    score,
    ndcg,
    utils,
    create_rankings,
    split,
    coverage_at_k,
    precision_at_k,
)


@pytest.fixture
def small_split_dataset():
    users = random.choices(arange(100), k=500)
    items = random.choices(arange(100), k=500)
    users.extend([101, 102])
    items.extend([101, 102])
    data = c_[asarray(users), asarray(items)]
    df = DataFrame(data=data, columns=["user", "item"])
    df = df.drop_duplicates(["user", "item"])
    return df


def test_single_process_ndcg_success():
    n = 10000
    predicted = [list(range(10)) for _ in range(n)]
    actual = [list(range(random.randint(10, 30))) for _ in range(n)]

    assert sum(score(ndcg, actual, predicted, n_cpu=1)) == n


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

    df = df.drop_duplicates(subset=["user", "item"])

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

    users, items = create_rankings(test_dataset, train_dataset)

    a, b, _ = dataset.to_components()
    all_users, all_items = groupby(a, b)

    for i in range(len(items)):
        # items includes no more than one element from 'all_items'.
        assert len(intersect1d(items[i], all_items[i])) == 1


def test_split_min_records(small_split_dataset):
    train, test = utils.split(small_split_dataset, frac_train=0.75, min_records=2)

    # users '101' and '102' appear once, check they don't exist
    assert (train["user"] == 101).sum() == 0
    assert (train["user"] == 102).sum() == 0


def test_split_ignore_users(small_split_dataset):
    frac = 0.5
    train, test = utils.split(
        small_split_dataset, frac_train=0.75, frac_ignored_users=frac
    )

    df = train.append(test)
    n_sampled_users = df["user"].nunique()
    n_users = small_split_dataset["user"].nunique()
    assert n_sampled_users <= int(math.ceil(frac * n_users))


def test_score_throws_error_for_mismatched_inputs():
    a = ones((10, 100), dtype=int)
    b = ones((9, 100), dtype=int)

    with pytest.raises(ValueError):
        score(ndcg, a, b)


def test_coverage_at_k_correctness():
    a = arange(0, 1000)
    b = a[:100]

    cov = coverage_at_k(a.reshape(-1, 1), b.reshape(-1, 1), k=10)

    assert cov == 0.1


def test_precision_at_k_correctness():
    options = arange(0, 1000)

    actual = []
    predicted = []
    for i in range(100):
        a = random.choices(options, k=6)
        p = random.choices(a, k=3)
        p_ = random.choices(options[~isin(options, a)], k=3)
        p.extend(p_)
        actual.append(a)
        predicted.append(p)

    assert score(precision_at_k, actual, predicted).mean() == 0.5


def test_fold_correctness():
    data = [
        ["jane smith", "london", "doctor"],
        ["dave davey", "manchester", "spaceman"],
        ["john appleseed", "san francisco", "corporate shill"],
        ["jenny boo", "paris", "ninja"],
    ]

    raw_meta = DataFrame(data=data, columns=["user", "location", "occupation"])

    meta = fold(
        raw_meta, "user", ["location", "occupation"]
    )

    assert meta.shape[1] == 2
    assert meta.shape[0] == len(asarray(data)[:, 1:].flatten())

    for element in data:
        tags = meta[meta["user"] == element[0]]["tag"].values.tolist()
        assert all(_ in tags for _ in element[1:])
