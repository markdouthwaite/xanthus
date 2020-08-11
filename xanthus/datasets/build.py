"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Any, Tuple

from pandas import DataFrame

from .core import Dataset, DatasetEncoder
from . import utils
from ..evaluate import leave_one_out, split


def build(
    df: DataFrame,
    user_df: Optional[DataFrame] = None,
    item_df: Optional[DataFrame] = None,
    policy: str = "leave_one_out",
    **kwargs: Optional[Any]
) -> Tuple[Dataset, Dataset]:
    """
    Utility function for building train and test datasets for recommender models.

    Parameters
    ----------
    df: DataFrame
        The input dataset. This must have the fields 'user', 'item' and may optionally
        include the field 'rating'.
    item_df: DataFrame, optional
        Optionally provide a DataFrame containing item metadata. This frame should
        include the fields 'item' and 'tag', where 'tag' corresponds to some metadata
        value associated with that item (e.g. genre, director, language in a movie
        use-case).
    user_df: DataFrame, optional
        Optionally provide a DataFrame containing user metadata. This frame should
        include the fields 'user' and 'tag', where 'tag' corresponds to some metadata
        value associated with that user (e.g. age group, location, language in a movie
        use-case).
    policy: str
        The 'split' policy to use when creating 'train' and 'test' sets. By default,
        'leave_one_out' is used. Otherwise, a custom split is used.
    kwargs: optional
        Additional arguments to be passed to a custom 'split' method.

    Returns
    -------
    output: tuple
        A pair where the first element corresponds to a configured 'train' dataset, and
        the second corresponds to a configured 'test' dataset.

    See Also
    --------
    xanthus.evaluate.split

    """

    # split the data into train and test sets
    if policy == "leave_one_out":
        train, test = leave_one_out(df)
    else:
        train, test = split(df, **kwargs)

    # extract item tags (if they're available)
    if item_df is not None:
        item_tags = item_df["tag"]
        items = df["item"].append(item_df["item"])
    else:
        item_tags = None
        items = df["item"]

    # extract user tags (if they're available)
    if user_df is not None:
        user_tags = user_df["tag"]
        users = df["user"].append(user_df["user"])
    else:
        user_tags = None
        users = df["user"]

    # fit the encoder.
    encoder = DatasetEncoder()
    encoder.fit(users=users, items=items, user_tags=user_tags, item_tags=item_tags)

    # build the train dataset.
    train = Dataset.from_df(
        train, encoder=encoder, normalize=utils.as_implicit, item_meta=item_df
    )

    # build the test dataset.
    test = Dataset.from_df(
        test, encoder=encoder, normalize=utils.as_implicit, item_meta=item_df
    )

    # and that's it, you're done!
    return train, test
