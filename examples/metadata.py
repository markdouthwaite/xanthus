"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import fire

import numpy as np
import pandas as pd

from xanthus import datasets
from xanthus.models import GeneralizedMatrixFactorization, utils
from xanthus.evaluate import create_rankings, score, metrics

np.random.seed(42)


def run(version="ml-latest-small", samples=4, input_dim=3, batch_size=256, epochs=1):

    # download the dataset
    datasets.movielens.download(version=version)

    # load and prepare datasets
    df = pd.read_csv(f"data/{version}/ratings.csv")

    item_df = pd.read_csv(f"data/{version}/movies.csv")
    item_df = item_df.rename(columns={"movieId": "item"})

    item_df = datasets.utils.fold(
        item_df, "item", ["genres"], fn=lambda s: (t.lower() for t in s.split("|"))
    )

    df = df.rename(columns={"userId": "user", "movieId": "item"})

    train_dataset, test_dataset = datasets.build(
        df, item_df=item_df, policy="leave_one_out"
    )

    n, m = train_dataset.user_dim, train_dataset.item_dim

    # build training arrays (you can also use 'batched' models too.
    users_x, items_x, y = train_dataset.to_components(
        negative_samples=samples, output_dim=input_dim
    )

    # initialize, compile and train the model
    model = GeneralizedMatrixFactorization(n, m)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit([users_x, items_x], y, epochs=epochs)

    # evaluate the model as described in He et al.
    users, items = create_rankings(
        test_dataset, train_dataset, output_dim=input_dim, n_samples=100, unravel=True
    )
    _, test_items, _ = test_dataset.to_components(shuffle=False)

    scores = model.predict([users, items], verbose=1, batch_size=batch_size)
    recommended = utils.reshape_recommended(users, items, scores, 10, mode="array")

    ndcg = score(metrics.truncated_ndcg, test_items, recommended).mean()
    hr = score(metrics.hit_ratio, test_items, recommended).mean()

    print(f"NDCG={ndcg}, HitRatio={hr}")


if __name__ == "__main__":
    fire.Fire(run)
