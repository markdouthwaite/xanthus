import numpy as np
import pandas as pd

from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel
from xanthus.dataset import Dataset, DatasetEncoder
from xanthus.evaluate import split, score, he_sampling, metrics


# test first column of he_sample matches leave-one-out items.


df = pd.read_csv("data/movielens-100k/ratings.csv",)

df = df.rename(columns={"userId": "user", "movieId": "item"})

encoder = DatasetEncoder()
encoder.partial_fit(df["user"], df["item"])

train, test = split(df, n_test=1)

dataset = Dataset.from_df(df, encoder=encoder, normalize=lambda _: np.ones_like(_))

train_dataset = Dataset.from_df(
    train, encoder=encoder, normalize=lambda _: np.ones_like(_)
)
test_dataset = Dataset.from_df(
    test, encoder=encoder, normalize=lambda _: np.ones_like(_)
)

_, test_items, _ = test_dataset.to_arrays(shuffle=False)

users, items = he_sampling(test_dataset, train_dataset)

model = GMFModel(fit_params=dict(epochs=5, batch_size=256), n_factors=64, negative_samples=4)
model.fit(train_dataset)

recommended = model.predict(test_dataset, users=users, items=items, n=10)

print(score(metrics.ndcg, test_items, recommended, k=1).mean())
print(score(metrics.precision_at_k, test_items, recommended, k=10).mean())
