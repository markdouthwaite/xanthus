"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import warnings
from typing import Tuple

from pandas import DataFrame, concat

from numpy import (
    in1d,
    concatenate,
    ndarray,
    unique,
    c_,
    asarray,
)
from numpy.random import choice

from ..datasets import Dataset, groupby


def _ignore(df: DataFrame, elements: ndarray, key: str, frac: float) -> DataFrame:
    """
    Remove a random fraction 'frac' of 'elements' with 'key' from a dataframe 'df'.
    """

    ignored_items_size = int(len(elements) * frac)
    ignored_items = choice(elements, size=ignored_items_size, replace=False)
    df = df[~df[key].isin(ignored_items)]
    return df


def leave_one_out(
    df,
    shuffle=True,
    frac_ignored_users: float = 0.0,
    frac_ignored_items: float = 0.0,
    deduplicate: bool = True,
) -> Tuple[DataFrame, ...]:
    """
    Wraps the 'split' function to produce leave-one-out evaluation protocol.
    """

    return split(
        df,
        shuffle=shuffle,
        frac_ignored_users=frac_ignored_users,
        frac_ignored_items=frac_ignored_items,
        deduplicate=deduplicate,
    )


def split(
    df: DataFrame,
    frac_train: float = 0.0,
    shuffle: bool = True,
    n_test: int = 1,
    min_records: int = 1,
    frac_ignored_users: float = 0.0,
    frac_ignored_items: float = 0.0,
    deduplicate: bool = True,
) -> Tuple[DataFrame, ...]:
    """
    A recommendation-system focused train-test split utility function.

    This function was inspired by the Azure ML Studio 'recommender split' utility [1].

    Using default parameters, this function implements the 'leave-one-out' sampling
    approach, whereby a single record from each user is withheld for evaluation
    purposes, and all others included in the training set.

    Parameters
    ----------
    df: DataFrame
        An input dateframe. This frame should represent some set of interactions between
        some 'user' set, and an 'item' set. The fields 'user' and 'item' _must_ be
        present in the frame. As in any CF problem setup, these interactions could be
        page views, purchases, email opens etc.
    frac_train: float
        The fraction of users in the dataset to be used exclusively in the training set.
        None of these users will appear in the test set. The remaining fraction of users
        will be considered 'test candidates', and will be sampled to create the output
        test set, and users in this 'test candidate' set may appear in _both_ the train
        and test set. By default, all users will appear in the test set (frac_train=0).
    shuffle: bool
        Indicate whether the dataset should be shuffled.
    n_test: int
        The number of test samples (from the test candidates) to be sampled. For
        example, if you set `frac_train` to 0.75, 25% of all interactions will be held
        out as test candidates. The function will then sample `n_test` interactions from
        this set. These samples will be come the test set. All remaining candidate test
        samples are appended to the training set. Note that if this is set too high,
        you may have very few 'eligible' users in your candidate set to sample. If there
        are no eligible users, this will raise an error.
    min_records: int
        The minimum number of interactions a user must have had in order to appear in
        the output training data.
    frac_ignored_users: float
        An approximate fraction of users to be ignored. This can be useful to reduce the
        size of large datasets for training purposes.
    frac_ignored_items: float
        An approximate fraction of items to be ignored. This can be useful to reduce the
        size of large datasets for training purposes.
    deduplicate: bool
        Indicate whether to deduplicate interactions (i.e. interactions with the same
        'user' and 'item'). Default is 'True'.

    Returns
    -------
    output: (DataFrame, DataFrame)
        Your specified 'train' and 'test' dataframes.

    References
    ----------
    [1] Inspired by https://bit.ly/3fCL9Xc (Azure Recommender Split)

    Notes
    -----
    * Benchmarks:
      (0 on 1m records, 500k users, 10k items - ~5s
      (1 on 10m records, 5m users, 10k items - ~60s
      (2 on 20m records, 10m users, 10k items - ~150s
    * Scales roughly linearly in the number of records.

    """

    if shuffle:
        df = df.sample(frac=1.0)

    columns = df.columns

    df["_user"] = df["user"].astype("category").cat.codes
    df["_item"] = df["item"].astype("category").cat.codes

    if deduplicate:
        df = df.drop_duplicates(["_user", "_item"])

    unique_items = df["_item"].unique()
    unique_users = df["_user"].unique()

    if frac_ignored_items > 0:
        df = _ignore(df, unique_items, "_item", frac_ignored_items)

    if frac_ignored_users > 0:
        df = _ignore(df, unique_users, "_user", frac_ignored_users)

    if min_records > 1:
        df = df[df.groupby("_user")["_user"].transform("size") >= min_records]

    # build candidate test frame.
    training_users_size = int(len(unique_users) * frac_train)
    training_users = choice(unique_users, size=training_users_size, replace=False)
    candidate_test_users = unique_users[~in1d(unique_users, training_users)]

    candidate_test_df = df[df["_user"].isin(candidate_test_users)]

    grouped = candidate_test_df.groupby("_user").size() > n_test
    training_users = concatenate(
        [training_users, grouped.index.values[~grouped.values]]
    )
    eligible = grouped.index.values[grouped.values]

    if len(eligible) == 0:
        raise ValueError(
            f"Could not find users in the test sample with more than {n_test} records."
            "Make sure your "
        )

    if len(eligible) < int(len(candidate_test_users) * 0.75):
        warnings.warn(
            "The total number of users that can be sampled from for your test "
            "set is less than 75% of the available users. Consider reducing your"
            f"chosen 'n_test' (it is set to {n_test})."
        )

    # filter the candidate test frame.
    candidate_test_df = candidate_test_df[candidate_test_df["_user"].isin(eligible)]
    excluded_test_df = candidate_test_df[~candidate_test_df["_user"].isin(eligible)]

    # create the training set.
    training_users = concatenate([training_users, excluded_test_df["_user"].unique()])

    train_df = df[df["_user"].isin(training_users)]

    # sample from the candidate test frame.
    test_df_grouped = candidate_test_df.groupby("_user", group_keys=False)
    candidate_test_df["_group"] = test_df_grouped.cumcount()

    test_df = candidate_test_df[candidate_test_df["_group"] < n_test]
    aux_train_df = candidate_test_df[candidate_test_df["_group"] >= n_test]

    # Add interactions that are not in the sampled test frame back to the training set.
    train_df = concat([train_df, aux_train_df])

    return train_df[columns], test_df[columns]


def create_rankings(
    a: Dataset, b: Dataset, n_samples: int = 100, unravel: bool = False, **kwargs: int
) -> Tuple[ndarray, ndarray]:
    """
    Sample a dataset 'a' with 'n' negative samples given interactions in dataset 'a'
    and 'b'.

    Practically, this function allows you to generate evaluation data as described in
    the work of He et al. [1]. The evaluation procedure assumes that the input datasets
    'a' and 'b' have been generated with a leave 'n' out policy, such that dataset 'b'
    corresponds to the 'training' dataset (i.e. dataset with 'left out' samples removed)
    and 'a' corresponds to the 'test' dataset with 'n' for each user with
    n_interactions > n. For each user in 'a', the function will return that user's 'n'
    left-out interactions, plus 'n_samples' negative samples (items the user has not
    interacted with in both the 'train' and 'test' datasets).

    Parameters
    ----------
    a: Dataset
        The 'test' dataset (the dataset you wish to use for evaluation).
    b: Dataset
        The 'train' dataset (the dataset you wish to include for purposes of sampling
        items the user has not interacted with -- negative samples).
    n_samples: int
        The total number of negative samples per user to generate. For example, if the
        dataset 'a' was generated from a leave-one-out split, and n_samples=100, that
        user would receive 101 samples.
    unravel: bool
        If 'True', the function will return two arrays, where the first element of the
        first array corresponds to the user _vector_ (i.e. user ID + optional metadata),
        the first element of the first array corresponds to an associated sampled item
        vector(i.e. item ID + optional metadata).

    Returns
    -------
    output: (ndarray, List[ndarray])
        If 'unravel=False', the first element corresponds to an array of _ordered_ user
        ids, the second the `n_samples+1`per-user samples.
        If `unravel=True`, the first element corresponds to an array of _ordered_ user
        vectors, the second to each individual item vector. See `unravel` argument and
        `_unravel_ranked`, below. This function is provided for use when evaluating
        Keras Models with the `predict` method.

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569

    """

    users, items, _ = a.to_components(
        negative_samples=n_samples,
        aux_matrix=b.interactions.tocsr(),
        shuffle=False,
        sampling_mode="absolute",
    )

    unique_users = unique(users)

    sampled_users, sampled_items = (
        users[len(unique_users) :],
        items[len(unique_users) :],
    )

    groups, grouped = groupby(sampled_users, sampled_items)

    grouped = c_[grouped, items[: len(unique_users)]]

    if unravel:
        return _unravel_sampled(unique_users, grouped, a, **kwargs)
    else:
        return unique_users, grouped


def _unravel_sampled(
    users: ndarray, ranked: ndarray, a: Dataset, output_dim: int = 1
) -> Tuple[ndarray, ndarray]:
    """
    Unravel two arrays of the form:
        user_{i}, [item_{i,0}, ..., item_{i, j}
    Into the form:
        user_{i}, item_{i, 0}
        ...
        user_{i}, item_{i, j}
    """

    z = list([user, item] for i, user in enumerate(users) for item in ranked[i])
    z = asarray(z)

    if output_dim > 1:
        users = asarray(list(a.iter_user(z[:, 0], n_dim=output_dim)))
        items = asarray(list(a.iter_item(z[:, 1], n_dim=output_dim)))
    else:
        users = z[:, 0]
        items = z[:, 1]

    return users, items
