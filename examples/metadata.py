"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import numpy as np
import pandas as pd

from xanthus.datasets import utils
from xanthus.models import neural
from xanthus.evaluate import he_sampling, score, metrics
from xanthus.utils import create_datasets

np.random.seed(42)

df = pd.read_csv("../data/movielens-100k/ratings.csv")

item_df = pd.read_csv("../data/movielens-100k/movies.csv")
item_df = item_df.rename(columns={"movieId": "item"})

item_df = utils.fold(
    item_df, "item", ["genres"], fn=lambda s: (t.lower() for t in s.split("|"))
)

df = df.rename(columns={"userId": "user", "movieId": "item"})

train_dataset, test_dataset = create_datasets(
    df, item_df=item_df, policy="leave_one_out"
)

_, test_items, _ = test_dataset.to_components(shuffle=False)

model = neural.GeneralizedMatrixFactorizationModel(
    fit_params=dict(epochs=1, batch_size=256), n_factors=8, negative_samples=4, n_meta=2
)

model.fit(train_dataset)

users, items = he_sampling(test_dataset, train_dataset)

recommended = model.predict(test_dataset, users=users, items=items, n=10)

print("t-nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.hit_ratio, test_items, recommended).mean())
