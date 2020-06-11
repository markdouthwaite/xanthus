from typing import List, Tuple

import fire

import numpy as np
from numpy import ndarray
import pandas as pd

from xanthus.evaluate import split, score, metrics, he_sampling
from xanthus.dataset import DatasetEncoder, Dataset

np.random.seed(42)


def main():
    # df = pd.read_csv("data/movielens-1m/ratings.dat", delimiter="::", names=["userId", "movieId", "rating", "timestamp"],)
    df = pd.read_csv("data/movielens-100k/ratings.csv")

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    encoder = DatasetEncoder()
    encoder.partial_fit(df["user"], df["item"])

    train, test = split(df, n_test=1)

    train_dataset = Dataset.from_frame(train, encoder=encoder)
    test_dataset = Dataset.from_frame(test, encoder=encoder)

    users, items = he_sampling(test_dataset, train_dataset, n_samples=100)


if __name__ == "__main__":
    fire.Fire(main)
