import fire

import numpy as np
import pandas as pd

from xanthus.evaluate import split, score, metrics
from xanthus.dataset import DatasetEncoder, Dataset
from xanthus.models import MatrixFactorizationModel

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

    train_dataset = Dataset.from_frame(train, encoder=encoder)
    test_dataset = Dataset.from_frame(test, encoder=encoder)

    model = MatrixFactorizationModel(factors=100,
                                     method="bpr",
                                     verify_negative_samples=True)
    model.fit(train_dataset)

    recommended = model.predict(test_dataset.users, n=k).tolist()

    print(metrics.coverage_at_k(test_dataset.all_items, recommended, k=k))
    print(score(metrics.pak, test_dataset.history, recommended, k=k).mean())
    print(score(metrics.ndcg, test_dataset.history, recommended, k=k).mean())


if __name__ == "__main__":
    fire.Fire(main)

"""
0.011619537275064267
0.051147540983606556
0.029008818712821727
"""