import os
import time
import numpy as np
import pandas as pd

from xanthus.datasets import Dataset, utils


print(os.path.split("hello/world/data.csv"))

# print(np.tile([0, 1, 2], (10, 1)))

# df = pd.read_csv("data/movielens-100k/ratings.csv")
# df = df.rename(columns={"userId": "user", "movieId": "item"})
#
# ds = Dataset.from_df(
#     df, normalize=utils.as_implicit
# )
#
# t0 = time.time()
# u, _, _ = ds.to_components()
# t1 = time.time()
# print(t1 - t0)
# print(len(u))
# d = (t1 - t0) / len(u)
