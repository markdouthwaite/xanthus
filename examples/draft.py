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


# user_x, item_x, y = train_dataset.to_arrays()
#
# x = pd.DataFrame(data=np.c_[user_x, item_x, y].astype(int),
#                  columns=["user", "item", "rating"])
#
# user_x, item_x, y = train_dataset.to_arrays(
#     negative_samples=1,
# )
#
# sx = pd.DataFrame(data=np.c_[user_x, item_x, y].astype(int),
#                   columns=["user", "item", "rating"])

# for name, group in x.groupby("user"):
#     print(group.shape)
#     negatives = sx[(sx["user"] == name) & (sx["rating"] == 0)]
#     positives = sx[(sx["user"] == name) & (sx["rating"] > 0)]
#     print(positives)
#     print(negatives)
#     print(negatives.shape)
#     print(positives.shape)
#     print(negatives.item.isin(positives.item).values.sum())
#     print((negatives.item < 1000).sum() / len(negatives.item))
#     break


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
        # for each epoch -> different sample!

        model = utils.get_mf_model(
            dataset.all_users.shape[0],
            dataset.all_items.shape[0],
            n_factors=self._n_factors,
            reg=self._l2_reg,
        )
        model.compile(optimizer=self._opt, loss=self._loss)
        epochs = self._kwargs.get("epochs", 1)
        params = {k: v for k, v in self._kwargs.items() if k != "epochs"}

        for i in range(epochs):
            user_x, item_x, y = dataset.to_arrays(negative_samples=self._neg,)

            tu, vu, ti, vi, ty, vy = train_test_split(
                user_x, item_x, y, test_size=self._val_split, shuffle=True
            )

            model.fit(
                [tu, ti],
                ty,
                epochs=i + 1,
                initial_epoch=i,
                **params,
                validation_data=([vu, vi], vy)
            )

        self._model = model
        return self

    def predict(self, users, dataset, n=10, items=None):
        recommended = []
        items = items if items is not None else dataset.all_items
        users = users if users is not None else dataset.users
        mat = dataset.interactions.tocsr()
        for i, user in enumerate(users):
            x = [np.asarray([user] * len(items)), items]
            h = self._model(x).numpy().flatten()
            ranked = h.argsort()[::-1]

            recommended.append(rank(h, n, r=items))

        return np.asarray(recommended)


def rank(w, n, encodings=None, excluded=None):
    ranked = w.argsort()[::-1]

    if encodings is not None:
        ranked = encodings[ranked]

    if excluded is not None:
        ranked = ranked[~np.isin(ranked, excluded)]

    return ranked[:n]


def main(k=10):
    w = np.asarray([0.1, 0.2, 0.4, 0.3])
    r = np.asarray([4, 9, 18, 22])
    f = np.asarray([4, 6])

    print(rank(w, 2))
    print(rank(w, 4, r))
    print(rank(w, 4, r, f))

    # df = pd.read_csv("data/movielens-100k/ratings.csv",)
    #
    # df = df.rename(columns={"userId": "user", "movieId": "item"})
    #
    # encoder = DatasetEncoder()
    # encoder.partial_fit(df["user"], df["item"])
    #
    # train, test = split(df, n_test=1)
    #
    # train_dataset = Dataset.from_df(
    #     train, encoder=encoder, normalize=lambda _: np.ones_like(_)
    # )
    # test_dataset = Dataset.from_df(
    #     test, encoder=encoder, normalize=lambda _: np.ones_like(_)
    # )
    #
    # users, items = he_sampling(test_dataset, train_dataset, n_samples=100)
    #
    # model = GMFModel(
    #     optimizer=Adam(lr=1e-3),
    #     loss=BinaryCrossentropy(),
    #     epochs=1,
    #     batch_size=64,
    #     n_factors=16,
    #     negatives_samples=5,
    # )
    #
    # model.fit(train_dataset)
    #
    # mat = train_dataset.interactions.tocsr()
    # recommended = model.predict(users, train_dataset, n=k)
    # for i, user in enumerate(users):
    #     print(recommended[i])
    #     print(mat[user].nonzero()[1])
    #     print("")

    # print(recommended)
    # print(metrics.coverage_at_k(test_dataset.all_items, recommended, k=k))
    # print(score(metrics.pak, test_dataset.history, recommended, k=2).mean())
    # print(score(metrics.ndcg, test_dataset.history, recommended, k=2).mean())


main()

# if __name__ == "__main__":
#     fire.Fire(main)
