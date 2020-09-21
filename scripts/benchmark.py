import fire
import logging

import pandas as pd
from xanthus.datasets.build import build
from xanthus.datasets.movielens import download

from xanthus.utils.benchmarking import (
    benchmark,
    save,
)

from managers import NeuralModelManager, BaselineModelManager


logging.basicConfig(level=logging.INFO)


def run(experiment="benchmarks-2", factors=(8,), epochs=15, root="data"):
    if isinstance(factors, (int, str)):
        factors = (int(factors),)

    download()

    df = pd.read_csv("data/ml-latest-small/ratings.csv")
    df = df.rename(columns={"movieId": "item", "userId": "user"})
    train, val = build(df)

    for factor in factors:
        managers = [
            BaselineModelManager("als", factors=factor, datasets=(train, val)),
            BaselineModelManager("bpr", factors=factor, datasets=(train, val)),
            NeuralModelManager("nmf", factors=factor, datasets=(train, val)),
            NeuralModelManager("gmf", factors=factor, datasets=(train, val)),
            NeuralModelManager("mlp", factors=factor, datasets=(train, val)),
        ]

        for manager in managers:
            results, info = benchmark(manager, epochs)
            save(experiment, manager, results, info, root=root, identifier=str(factor))


if __name__ == "__main__":
    fire.Fire(run)
