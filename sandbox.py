import numpy as np
import pandas as pd

from xanthus.dataset import Dataset, DatasetEncoder
from xanthus.evaluate import split


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


