import numpy as np
import pandas as pd

from xanthus.models.baseline import MatrixFactorizationModel as MFModel
from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel
from xanthus.datasets import Dataset, DatasetEncoder
from xanthus.evaluate import split, score, he_sampling, metrics


# test first column of he_sample matches leave-one-out items.

"""
import pandas as pd
from xanthus.datasets import Dataset
from xanthus.models.baseline import MatrixFactorizationModel as MFModel
from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel

df = pd.read_csv("...")
dataset = Dataset.from_df(df)

model = GMFModel()
model.fit(dataset)

dataset.to_dense()
dataset.encoder.to_df()
dataset.sampler.
r
recommended = model.predict(dataset)
"""

interactions = pd.read_csv("data/movielens-100k/ratings.csv")
interactions = interactions.rename(columns={"userId": "user", "movieId": "item"})
metadata = pd.read_csv("data/movielens-100k/movies.csv")

item = metadata["movieId"]
tags = metadata["genres"]
data = ([item[i], t] for i in range(len(item)) for t in tags[i].split("|"))

item_meta = pd.DataFrame(data=data, columns=["item", "tag"])

ds = Dataset.from_df(interactions, item_meta=item_meta)

for user, item, rating in ds.iter(output_dim=4, shuffle=False):
    print(user, item, rating)
    break

# print(df[["movieId", "genres"]])

# df = df.rename(columns={"userId": "user", "movieId": "item"})
#
# ds = Dataset.from_df(df)
#
# for user, item, rating in ds.iter(shuffle=False):
#     print(user, item, rating)
#     break


# df = pd.read_csv("data/movielens-100k/ratings.csv",)
#
# df = df.rename(columns={"userId": "user", "movieId": "item"})
#
# encoder = DatasetEncoder()
# encoder.partial_fit(df["user"], df["item"])
#
# dataset = Dataset.from_df(df, encoder=encoder, normalize=lambda _: np.ones_like(_))
# dataset.to_arr(negative_samples=4, sampling_mode="absolute")

# print(np.hstack([[[0], [1]], [[1, 2, 3], [2, 3, 4]]]))

# model = MFModel(iterations=5)
# model.fit(dataset)
# recommended = model.predict(dataset)
#
# print(recommended)
# print(dataset.encoder.inverse_transform(items=recommended[0]))

# train_dataset = Dataset.from_df(
#     train, encoder=encoder, normalize=lambda _: np.ones_like(_)
# )
# test_dataset = Dataset.from_df(
#     test, encoder=encoder, normalize=lambda _: np.ones_like(_)
# )
#
# _, test_items, _ = test_dataset.to_arrays(shuffle=False)
#
# users, items = he_sampling(test_dataset, train_dataset)
#
# model = GMFModel(fit_params=dict(epochs=5, batch_size=256), n_factors=64, negative_samples=4)
# model.fit(train_dataset)
#
# recommended = model.predict(test_dataset, users=users, items=items, n=10)
#
# print(score(metrics.ndcg, test_items, recommended, k=1).mean())
# print(score(metrics.precision_at_k, test_items, recommended, k=10).mean())
