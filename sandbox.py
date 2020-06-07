import time
import pandas as pd
import numpy as np
from xanthus.dataset import Dataset, DatasetEncoder


movies = pd.read_csv("data/movielens-1m/movies.dat", delimiter="::", names=["movieId", "title", "genres"])
interactions = pd.read_csv("data/movielens-1m/ratings.dat", delimiter="::", names=["userId", "movieId", "rating", "timestamp"])

new_dataset = []

for row in movies.to_dict(orient="records"):
    for genre in row["genres"].split("|"):
        new_dataset.append([row["movieId"], genre])

movies = pd.DataFrame(data=new_dataset, columns=["item", "tag"])
interactions = interactions.rename(columns={"userId": "user", "movieId": "item"})


# encoder = DatasetEncoder()
#
# encoder.fit(interactions["user"], interactions["item"])
#
# train_interactions = interactions.iloc[:int(len(interactions)/2)]
# test_interactions = interactions.iloc[:int(len(interactions)/2)]
#
# train_dataset = Dataset.from_frame(train_interactions, encoder=encoder)
# print(train_dataset.interactions.shape)
#
# test_dataset = Dataset.from_frame(test_interactions, encoder=encoder)
# print(test_dataset.interactions.shape)

t0 = time.time()
dataset = Dataset.from_frame(interactions, item_meta=movies)
print(dataset.interactions.shape)

t1 = time.time()
print(dataset.to_arrays(output_dim=5, negative_samples=2))
t2 = time.time()

print("load", t1 - t0)
print("build", t2 - t1)


# users, items, ratings = dataset.to_arrays()
#
# print(users)
# print(items)
# print(ratings)
