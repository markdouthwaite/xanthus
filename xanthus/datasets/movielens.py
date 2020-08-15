"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import os
import io
import zipfile
import warnings
from typing import Tuple, Any, NoReturn, TypeVar

import requests
import pandas as pd

from .utils import fold
from .build import build


Dataset = TypeVar("Dataset")


def download(
    version: str = "ml-latest-small",
    base_url: str = "http://files.grouplens.org/datasets/movielens/{version}.zip",
    output_dir: str = "data",
    unzip: bool = True,
) -> NoReturn:
    """Download a given movielens dataset."""

    path = os.path.join(output_dir, f"ml-{version}")
    response = requests.get(base_url.format(version=version))

    if os.path.exists(path):
        warnings.warn(f"Dataset already exists on path '{path}'. Aborting download.")
    else:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if unzip:
            zipfile.ZipFile(io.BytesIO(response.content)).extractall(output_dir)
        else:
            with open(os.path.join(output_dir, f"ml-{version}"), "wb") as file:
                file.write(response.content)


def _load_100k(path: str, policy: str, **kwargs: Any) -> Tuple[Dataset, Dataset]:
    """Load the ml-latest-small dataset."""

    df = pd.read_csv(os.path.join(path, "ratings.csv"))

    item_df = pd.read_csv(os.path.join(path, "movies.csv"))
    item_df = item_df.rename(columns={"movieId": "item"})
    item_df = fold(
        item_df, "item", ["genres"], fn=lambda s: (t.lower() for t in s.split("|"))
    )

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    train_ds, test_ds = build(df, item_df=item_df, policy=policy, **kwargs)
    return train_ds, test_ds


def _load_1m(path: str, policy: str, **kwargs: Any) -> Tuple[Dataset, Dataset]:
    """Load the ml-1m dataset."""

    df = pd.read_csv(
        os.path.join(path, "ratings.dat"),
        names=["userId", "movieId", "rating", "timestamp"],
        delimiter="::",
        engine="python",
    )

    item_df = pd.read_csv(
        os.path.join(path, "movies.dat"),
        names=["movieId", "title", "genres"],
        delimiter="::",
        engine="python",
    )
    item_df = item_df.rename(columns={"movieId": "item"})
    item_df = fold(
        item_df, "item", ["genres"], fn=lambda s: (t.lower() for t in s.split("|"))
    )

    user_df = pd.read_csv(
        os.path.join(path, "users.dat"),
        names=["userId", "gender", "age", "job", "zip"],
        delimiter="::",
        engine="python",
    )
    user_df = user_df.rename(columns={"userId": "user"})
    user_df = user_df[["user", "gender", "age"]]
    user_df = fold(user_df, "user", ["gender", "age"])

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    train_ds, test_ds = build(
        df, user_df=user_df, item_df=item_df, policy=policy, **kwargs
    )
    return train_ds, test_ds


def load(
    dirname: str = "data",
    version: str = "ml-latest-small",
    policy: str = "leave_one_out",
    **kwargs: Any,
) -> Tuple[Dataset, Dataset]:
    """Load a chosen Movielens dataset as train and test datasets."""

    path = os.path.join(dirname, version)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find '{path}'. Try downloading the data first with the "
            f"`xanthus.datasets.movielens.download` function."
        )

    if version == "ml-latest-small":
        return _load_100k(path, policy, **kwargs)
    elif version == "ml-1m":
        return _load_1m(path, policy, **kwargs)
