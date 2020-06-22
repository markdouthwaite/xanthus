import fire

import numpy as np
import pandas as pd

from tensorflow.keras.optimizers import Adam, Adagrad
from tensorflow.keras.losses import BinaryCrossentropy

from xanthus.evaluate import split, score, metrics, he_sampling
from xanthus.dataset import DatasetEncoder, Dataset
from xanthus.models import utils

from sklearn.model_selection import train_test_split

np.random.seed(42)


def main(k=10):
    df = pd.read_csv("data/movielens-100k/ratings.csv",)

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    encoder = DatasetEncoder()
    encoder.partial_fit(df["user"], df["item"])

    train, test = split(df, n_test=1)

    train_dataset = Dataset.from_df(
        train, encoder=encoder, normalize=lambda _: np.ones_like(_)
    )
    test_dataset = Dataset.from_df(
        test, encoder=encoder, normalize=lambda _: np.ones_like(_)
    )

    users, items = he_sampling(test_dataset, train_dataset, n_samples=100)

    model = GMFModel(
        optimizer=Adam(lr=1e-3),
        loss=BinaryCrossentropy(),
        epochs=7,
        batch_size=64,
        n_factors=8,
    )

    model.fit(train_dataset)

    recommended = model.predict(users, train_dataset, items=items)

    # print(metrics.coverage_at_k(test_dataset.all_items, recommended, k=k))
    print(score(metrics.pak, test_dataset.history, recommended, k=k).mean())
    print(score(metrics.ndcg, test_dataset.history, recommended, k=k).mean())


if __name__ == "__main__":
    fire.Fire(main)
