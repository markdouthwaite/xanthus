import numpy as np
import pandas as pd
from xanthus.datasets import Dataset, DatasetEncoder
# from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel
from xanthus.models.baseline import MatrixFactorizationModel as MFModel
from xanthus.evaluate import he_sampling, score, metrics, leave_one_out


df = pd.read_csv("../data/movielens-100k/ratings.csv",)

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

model = MFModel(
    # fit_params=dict(epochs=10, batch_size=256), n_factors=64, negative_samples=4
)
model.fit(train_dataset)

users, items = he_sampling(test_dataset, train_dataset)
recommended = model.predict(test_dataset, users=users, items=items, n=10)

print(score(metrics.ndcg, test_items, recommended, k=1).mean())
print(score(metrics.precision_at_k, test_items, recommended, k=10).mean())

"""
# BPR - 100k @ 15
0.34918032786885245
0.7688524590163934

# ALS - 100k @ 15
0.39672131147540984
0.7688524590163934

# NeuMF - 100k @ 10
0.3836065573770492
0.8327868852459016

# GMF - 100k
0.37868852459016394
0.8508196721311475

# MLP - 100k @ 10 epochs
0.3655737704918033
0.8327868852459016
"""