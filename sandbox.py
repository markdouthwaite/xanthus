import time
import pandas as pd
from xanthus.dataset import Dataset, DatasetEncoder

movies = pd.read_csv("data/movies.csv")[["movieId", "genres"]]

new_dataset = []

for row in movies.to_dict(orient="records"):
    for genre in row["genres"].split("|"):
        new_dataset.append([row["movieId"], genre])

movies = pd.DataFrame(data=new_dataset, columns=["item", "tag"])

interactions = pd.read_csv("data/ratings.csv")
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

dataset = Dataset.from_frame(interactions, item_meta=movies)
print(dataset.interactions.shape)

t1 = time.time()
print(dataset.to_arrays(output_dim=5))
t2 = time.time()
print(t2 - t1)

dataset.to_txt("data/encoded/")


# users, items, ratings = dataset.to_arrays()
#
# print(users)
# print(items)
# print(ratings)
