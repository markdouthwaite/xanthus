import pandas as pd
from xanthus.datasets import utils


data = [
    ["jane smith", "london", "doctor"],
    ["dave davey", "manchester", "spaceman"],
    ["john appleseed", "san francisco", "corporate shill"],
    ["jenny boo", "paris", "ninja"],
]

# raw_meta = pd.DataFrame(data=data, columns=["user", "location", "occupation"])

raw_meta = pd.read_csv("data/movielens-100k/movies.csv")
raw_meta = raw_meta.rename(columns={"movieId": "item"})

meta = utils.fold(
    raw_meta, "item", ["genres"], fn=lambda s: (t.lower() for t in s.split("|"))
)

print(meta)
