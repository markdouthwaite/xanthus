"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import pytest

import numpy as np

from xanthus.datasets import DatasetEncoder


def test_fit_user_item_succeeds(sample_dataset):
    d = DatasetEncoder()
    assert d.fit(users=sample_dataset[:, 0], items=sample_dataset[:, 1]) == d


def test_fit_transform_user_item_succeeds(sample_dataset):
    d = DatasetEncoder()
    r = d.fit_transform(users=sample_dataset[:, 0], items=sample_dataset[:, 1])

    # check overall shape of responses matches
    assert r["users"].shape == sample_dataset[:, 0].shape
    assert r["items"].shape == sample_dataset[:, 1].shape

    # check number of unique elements matches.
    assert np.unique(r["users"]).shape == np.unique(sample_dataset[:, 0]).shape
    assert np.unique(r["items"]).shape == np.unique(sample_dataset[:, 1]).shape


def test_fit_reversible_user_item_transform(sample_dataset):
    users, items = sample_dataset[:, 0], sample_dataset[:, 1]

    d = DatasetEncoder()
    d.fit(users=users, items=items)

    r = d.inverse_transform(**d.transform(users=users, items=items))

    assert r["users"].shape == users.shape
    assert r["items"].shape == items.shape

    assert np.all(r["users"] == users)
    assert np.all(r["items"] == items)


def test_partial_fit_item_user_succeeds(sample_dataset):
    users, items = sample_dataset[:, 0], sample_dataset[:, 1]
    n = int(users.shape[0] / 2)

    a_users, a_items = users[:n], items[:n]
    b_users, b_items = users[n:], items[n:]

    d = DatasetEncoder()
    d.partial_fit(users=a_users, items=a_items)
    d.partial_fit(users=b_users, items=b_items)

    r = d.transform(users=users, items=items)

    assert r["users"].shape == users.shape
    assert r["items"].shape == items.shape

    assert np.unique(r["users"]).shape == np.unique(users).shape
    assert np.unique(r["items"]).shape == np.unique(items).shape


def test_transform_fails_for_unknown_elements(sample_dataset):
    users, items = sample_dataset[:, 0], sample_dataset[:, 1]
    n = int(users.shape[0] / 4)

    with pytest.raises(KeyError):
        a_users, a_items = users[:n], items[:n]
        b_users, b_items = users[n:], items[n:]

        d = DatasetEncoder()
        d.fit(users=a_users, items=a_items)

        d.transform(users=b_users, items=b_items)
