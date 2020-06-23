from typing import Optional, Any

import fire

import numpy as np
import pandas as pd

from xanthus.dataset import DatasetEncoder, Dataset
from xanthus.models import MatrixFactorizationModel as MFModel
from xanthus.evaluate import split, score, metrics, he_sampling


np.random.seed(42)


def main(
    k: int = 10,
    method: str = "als",
    iterations: int = 25,
    factors: int = 64,
    **kwargs: Optional[Any]
) -> None:

    df = pd.read_csv(
        "data/movielens-1m/ratings.dat",
        names=["userId", "movieId", "rating", "timestamp"],
        delimiter="::",
    )

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

    _, test_items, _ = test_dataset.to_numpy(shuffle=False)

    users, items = he_sampling(test_dataset, train_dataset)

    model = MFModel(method=method, iterations=iterations, factors=factors, **kwargs)
    model.fit(train_dataset)

    recommended = model.predict(test_dataset, users=users, items=items, n=k)

    print(score(metrics.ndcg, test_items, recommended, k=1).mean())
    print(score(metrics.precision_at_k, test_items, recommended, k=k).mean())


if __name__ == "__main__":
    fire.Fire(main)

    """
    0.3737704918032787
    0.8245901639344262
    """
