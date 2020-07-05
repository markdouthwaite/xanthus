import numpy as np
import pandas as pd
from tensorflow.keras import callbacks
from xanthus.datasets import Dataset, DatasetEncoder
from xanthus.models import NeuralMatrixFactorizationModel as GMFModel
from xanthus.models.baseline import MatrixFactorizationModel as MFModel
from xanthus.evaluate import he_sampling, score, metrics, leave_one_out


tensorboard_callback = callbacks.TensorBoard(log_dir='./logs', profile_batch=5)


# df = pd.read_csv(
#     "../data/movielens-1m/ratings.dat",
#     names=["userId", "movieId", "rating", "timestamp"],
#     delimiter="::",
# )

df = pd.read_csv("../data/movielens-100k/ratings.csv")

df = df.rename(columns={"userId": "user", "movieId": "item"})
print("loaded")
print(df.head())

train, test = leave_one_out(df)
print("split")

encoder = DatasetEncoder()
encoder.partial_fit(df["user"], df["item"])

print("encoded")

train_dataset = Dataset.from_df(
    train, encoder=encoder, normalize=lambda _: np.ones_like(_)
)

print("got training set")

test_dataset = Dataset.from_df(
    test, encoder=encoder, normalize=lambda _: np.ones_like(_)
)

print("got test set")

_, test_items, _ = test_dataset.to_components(shuffle=False)

print("get test items")

# model = GMFModel(
#     fit_params=dict(epochs=15, batch_size=256), n_factors=64, negative_samples=4
# )

print("got model")
model = MFModel(method="als", factors=64, iterations=15)

model.fit(train_dataset)

print("finished training")

users, items = he_sampling(test_dataset, train_dataset)

print("got sampled users")

recommended = model.predict(test_dataset, users=users, items=items, n=10)

print("got recommendations, scoring...")

print("nDCG", score(metrics.ndcg, test_items, recommended, k=1).mean())
print("t-nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.hit_ratio, test_items, recommended).mean())
