import fire

import numpy as np
import pandas as pd

from xanthus.evaluate import split, score, metrics
from xanthus.dataset import DatasetEncoder, Dataset
from xanthus.models import PopRankModel

np.random.seed(42)


def main(k: int = 10, sampling: str = "uniform", alpha: float = 10.0) -> None:
    df = pd.read_csv(
        "../data/movielens-1m/ratings.dat",
        names=["userId", "movieId", "rating", "timestamp"],
        delimiter="::",
    )

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    encoder = DatasetEncoder()
    encoder.partial_fit(df["user"].values, df["item"].values)

    train, test = split(df, frac_train=0.75, n_test=10)

    train_dataset = Dataset.from_df(train, encoder=encoder)
    test_dataset = Dataset.from_df(test, encoder=encoder)

    model = PopRankModel(limit=k, sampling=sampling, alpha=alpha)
    model.fit(train_dataset)

    recommended = model.predict(test_dataset.users, n=k).tolist()

    print(metrics.coverage_at_k(test_dataset.all_items, recommended, k=k))
    print(score(metrics.pak, test_dataset.history, recommended, k=k).mean())
    print(score(metrics.ndcg, test_dataset.history, recommended, k=k).mean())


if __name__ == "__main__":
    fire.Fire(main)