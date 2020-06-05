import time
import pandas as pd
from xanthus.dataset import Dataset

movies = pd.read_csv("movies.csv")[["movieId", "genres"]]

new_dataset = []

for row in movies.to_dict(orient="records"):
    for genre in row["genres"].split("|"):
        new_dataset.append([row["movieId"], genre])


movies = pd.DataFrame(data=new_dataset, columns=["item", "tag"])

interactions = pd.read_csv("ratings.csv")
interactions = interactions.rename(columns={"userId": "user", "movieId": "item"})

dataset = Dataset.from_frame(interactions, item_meta=movies)

t1 = time.time()
print(dataset.to_arrays(output_dim=5, negative_samples=5))
t2 = time.time()
print(t2 - t1)

# users, items, ratings = dataset.to_arrays()
#
# print(users)
# print(items)
# print(ratings)
