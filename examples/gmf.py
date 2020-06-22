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


class GMFModel:
    def __init__(
        self,
        loss=BinaryCrossentropy(),
        optimizer=Adam(lr=1e-4),
        n_factors=8,
        val_split=0.1,
        negatives_samples=4,
        **kwargs
    ):
        self._model = None
        self._kwargs = kwargs
        self._loss = loss
        self._opt = optimizer
        self._neg = negatives_samples
        self._val_split = val_split
        self._n_factors = n_factors
        self._l2_reg = 1e-6

    def fit(self, dataset):
        user_x, item_x, y = dataset.to_arrays(negative_samples=self._neg)
        tu, vu, ti, vi, ty, vy = train_test_split(
            user_x, item_x, y, test_size=self._val_split
        )
        model = utils.get_mf_model(
            dataset.all_users.shape[0],
            dataset.all_items.shape[0],
            n_factors=self._n_factors,
            reg=self._l2_reg,
        )
        model.compile(optimizer=self._opt, loss=self._loss)
        model.fit([tu, ti], ty, **self._kwargs, validation_data=([vu, vi], vy))

        self._model = model
        return self

    def predict(self, users, dataset, n=6, items=None):
        recommended = []
        items = items if items is not None else dataset.all_items
        users = users if users is not None else dataset.users
        mat = dataset.interactions.tocsr()
        for i, user in enumerate(dataset.users):
            x = [np.asarray([user] * len(items)), items]
            print(x)
            h = self._model(x).numpy().flatten()
            ranked = h.argsort()[::-1]
            ranked = ranked[~np.isin(ranked, mat[user].nonzero()[1])]
            recommended.append(ranked[:n])
            print(i + 1, "of", len(users))
        return np.asarray(recommended)


def main(k=10):
    df = pd.read_csv("data/movielens-100k/ratings.csv",)

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    encoder = DatasetEncoder()
    encoder.partial_fit(df["user"], df["item"])

    train, test = split(df, n_test=1)

    train_dataset = Dataset.from_df(train, encoder=encoder, normalize=lambda _: np.ones_like(_))
    test_dataset = Dataset.from_df(test, encoder=encoder, normalize=lambda _: np.ones_like(_))

    users, items = he_sampling(test_dataset, train_dataset, n_samples=100)

    model = GMFModel(optimizer=Adam(lr=1e-3),
                     loss=BinaryCrossentropy(),
                     epochs=7, batch_size=64,
                     n_factors=8)

    model.fit(train_dataset)

    recommended = model.predict(users, train_dataset, items=items)

    # print(metrics.coverage_at_k(test_dataset.all_items, recommended, k=k))
    print(score(metrics.pak, test_dataset.history, recommended, k=k).mean())
    print(score(metrics.ndcg, test_dataset.history, recommended, k=k).mean())


if __name__ == "__main__":
    fire.Fire(main)
