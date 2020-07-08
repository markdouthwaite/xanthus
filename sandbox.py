import pandas as pd
from xanthus.datasets import utils, DatasetEncoder


users = ["jane.smith@email.com", "john.appleseed@email.com"]
items = ["action-movie-0001", "action-movie-0002"]
encoder = DatasetEncoder()
encoded = encoder.fit_transform(users=users, items=items)
output = encoder.to_df(encoded["users"], [encoded["items"] for _ in encoded["users"]])
print(output)

# data = [
#     ["jane smith", "london", "doctor"],
#     ["dave davey", "manchester", "spaceman"],
#     ["john appleseed", "san francisco", "corporate shill"],
#     ["jenny boo", "paris", "ninja"],
# ]
#
# # raw_meta = pd.DataFrame(data=data, columns=["user", "location", "occupation"])
#
# raw_meta = pd.read_csv("data/movielens-100k/movies.csv")
# raw_meta = raw_meta.rename(columns={"movieId": "item"})
#
# meta = utils.fold(
#     raw_meta, "item", ["genres"], fn=lambda s: (t.lower() for t in s.split("|"))
# )
#
# print(meta)
