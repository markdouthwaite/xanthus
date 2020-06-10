import warnings
from typing import Tuple

from pandas import DataFrame, concat

from numpy import in1d, concatenate, ndarray
from numpy.random import choice


def _ignore(df: DataFrame, elements: ndarray, key: str, frac: float) -> DataFrame:
    """
    Remove a random fraction 'frac' of 'elements' with 'key' from a dataframe 'df'.
    """

    ignored_items_size = int(len(elements) * frac)
    ignored_items = choice(elements, size=ignored_items_size, replace=False)
    df = df[~df[key].isin(ignored_items)]
    return df


def split(
    df: DataFrame,
    frac_train: float,
    shuffle: bool = True,
    n_test: int = 1,
    min_records: int = 1,
    frac_ignored_users: float = 0.0,
    frac_ignored_items: float = 0.0,
    deduplicate: bool = True,
) -> Tuple[DataFrame, ...]:
    """
    A recommendation system focused train-test split utility function.

    This function was inspired by the Azure ML Studio 'recommender split' utility [1].

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
        and test set.
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
        The fraction of users to be ignored. This can be useful to reduce the size of
        large datasets for training purposes.
    frac_ignored_items: float
        The fraction of items to be ignored. This can be useful to reduce the size of
        large datasets for training purposes.
    deduplicate: bool
        Indicate whether to deduplicate interactions (i.e. interactions with the same
        'user' and 'item'). Default is 'True'.

    Returns
    -------
    output: (DataFrame, DataFrame)
        Your specified 'train' and 'test' dataframes.

    References
    ----------
    [1] Inspired by https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/split-data-using-recommender-split#:~:text=The%20Recommender%20Split%20option%20is,user%2Ditem%2Drating%20triples.

    Notes
    -----
    * Benchmarks:
      (0 on 1m records, 500k users, 10k items - ~3s
      (1 on 10m records, 5m users, 10k items - ~58s
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
