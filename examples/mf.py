from typing import List, Tuple

import fire

import numpy as np
import pandas as pd

from xanthus.evaluate import split, score, metrics, he_sampling
from xanthus.dataset import DatasetEncoder, Dataset
from xanthus.models import MatrixFactorizationModel as MFModel, PopRankModel

np.random.seed(40)


def normalize(arr):
    scaled = 0.9 * (arr - arr.min()) / (arr.max() - arr.min())
    scaled += 0.1
    return scaled


def main(k=10):
    # df = pd.read_csv("data/movielens-100k/ratings.csv")

    df = pd.read_csv(
        "data/movielens-1m/ratings.dat",
        names=["userId", "movieId", "rating", "timestamp"],
        delimiter="::",
    )

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    encoder = DatasetEncoder()
    encoder.partial_fit(df["user"], df["item"])

    train, test = split(df, n_test=1)

    train_dataset = Dataset.from_df(train, encoder=encoder, normalize=lambda _: np.ones_like(_))
    test_dataset = Dataset.from_df(test, encoder=encoder, normalize=lambda _: np.ones_like(_))

    users, items = he_sampling(test_dataset, train_dataset, n_samples=100)

    model = MFModel(method="bpr", factors=64, iterations=50)
    model.fit(train_dataset)
    recommended = model.predict(test_dataset, users=users, items=items, n=k)

    print(score(metrics.pak, test_dataset.history, recommended, k=10).mean())
    print(score(metrics.ndcg, test_dataset.history, recommended, k=1).mean())


if __name__ == "__main__":
    fire.Fire(main)
