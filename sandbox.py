import numpy as np
import pandas as pd
from xanthus.datasets import Dataset, DatasetEncoder
from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel
from xanthus.models.baseline import MatrixFactorizationModel as MFModel
from xanthus.evaluate import he_sampling, score, metrics, leave_one_out


df = pd.read_csv("data/movielens-100k/ratings.csv",)

df = df.rename(columns={"userId": "user", "movieId": "item"})

train, test = leave_one_out(df)

encoder = DatasetEncoder()
encoder.partial_fit(df["user"], df["item"])

train_dataset = Dataset.from_df(
    train, encoder=encoder, normalize=lambda _: np.ones_like(_)
)

test_dataset = Dataset.from_df(
    test, encoder=encoder, normalize=lambda _: np.ones_like(_)
)

_, test_items, _ = test_dataset.to_components(shuffle=False)

model = MFModel(method="bpr")

model.fit(train_dataset)

users, items = he_sampling(test_dataset, train_dataset)
recommended = model.predict(test_dataset, users=users, items=items, n=10)

rdf = encoder.to_df(users, recommended, item_cols="r{0}")

print(rdf)

# print(score(metrics.ndcg, test_items, recommended, k=1).mean())
# print(score(metrics.truncated_ndcg, test_items, recommended).mean())
# print(score(metrics.hit_ratio, test_items, recommended).mean())
