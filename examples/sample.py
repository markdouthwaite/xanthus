import numpy as np
import pandas as pd
from xanthus.datasets import Dataset, DatasetEncoder
from xanthus.models import GeneralizedMatrixFactorizationModel as GMFModel
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

# model = GMFModel(
#     fit_params=dict(epochs=15, batch_size=256), n_factors=100, negative_samples=4
# )

model = MFModel(method="bpr")

model.fit(train_dataset)

users, items = he_sampling(test_dataset, train_dataset)
recommended = model.predict(test_dataset, users=users, items=items, n=10)

print(score(metrics.ndcg, test_items, recommended, k=1).mean())
print(score(metrics.truncated_ndcg, test_items, recommended).mean())
print(score(metrics.hit_ratio, test_items, recommended).mean())

"""
# ALS - 100k @ 15
0.44754098360655736
0.6126378461742324
0.7852459016393443

# BPR - 100k @ 15
0.37868852459016394
0.5577336479002039
0.7475409836065574

# GMF - 100k @ 15 
0.40491803278688526
0.6060944045188948
0.8262295081967214

# NeuMF - 100k @ 15
0.3885245901639344
0.5972951514089754
0.8180327868852459

"""