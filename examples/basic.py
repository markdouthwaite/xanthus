import numpy as np
import pandas as pd
from xanthus.datasets.build import build
from xanthus.models.baseline import MatrixFactorization as MFModel
from xanthus.evaluate import create_rankings, score, metrics

df = pd.read_csv("../data/ml-latest-small/ratings.csv")
df = df.rename(columns={"movieId": "item", "userId": "user"})
train, val = build(df)

users, items, _ = val.to_components(shuffle=False)
test_users, test_items = create_rankings(val, train)

model = MFModel(factors=32, iterations=15)
model.fit(train)

recommended = model.predict(val, users=users, items=test_items, n=10)

print(score(metrics.pak, items, recommended).mean())
