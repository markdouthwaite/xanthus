from xanthus.datasets import download

download.movielens(output_dir="test")

# import os
# import time
# import numpy as np
# import pandas as pd
#
# from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel
# from xanthus.datasets import Dataset, DatasetEncoder, utils
# from xanthus.evaluate import leave_one_out, score, metrics, he_sampling
#
#
# ratings = pd.read_csv("data/movielens-100k/ratings.csv")
# movies = pd.read_csv("data/movielens-100k/movies.csv")
# title_mapping = dict(zip(movies["movieId"], movies["title"]))
#
# ratings = ratings.rename(columns={"userId": "user", "movieId": "item"})
# ratings.loc[:, "item"] = ratings["item"].apply(lambda _: title_mapping[_])
#
# ratings = ratings[ratings["rating"] > 3.0]
#
# train_df, test_df = leave_one_out(ratings)
#
# encoder = DatasetEncoder()
# encoder.fit(df["user"], df["item"])
#
# train_ds = Dataset.from_df(train_df, normalize=utils.as_implicit, encoder=encoder)
# test_ds = Dataset.from_df(test_df, normalize=utils.as_implicit, encoder=encoder)
#
# model = GMFModel(
#     fit_params=dict(epochs=10, batch_size=256), n_factors=32, negative_samples=4
# )
#
# model.fit(train_ds)
#
# _, test_items, _ = test_ds.to_components(shuffle=False)
#
# # evaluate
# users, items = he_sampling(test_ds, train_ds, n_samples=200)
#
# recommended = model.predict(test_ds, users=users, items=items, n=10)
#
# print("t-nDCG", score(metrics.ndcg, test_items, recommended).mean())
# print("HR@k", score(metrics.precision_at_k, test_items, recommended).mean())
#
# # results
# recommended = model.predict(users=users, items=items[:, 1:], n=3)
# recommended_df = encoder.to_df(users, recommended)

recommended_df.to_csv("recs.csv", index=False)
