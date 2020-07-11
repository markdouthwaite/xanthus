"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import os
from typing import Optional, Any

import fire
import numpy as np
import pandas as pd
from tensorflow.keras import callbacks
from xanthus.models import baseline, neural
from xanthus.evaluate import he_sampling, score, metrics
from xanthus.utils import create_datasets


np.random.seed(42)


def _run_trials(
    models,
    configs,
    train,
    test,
    sampled_users,
    sampled_items,
    held_out_items,
    n_trials=3,
):
    results = []

    for i in range(n_trials):
        for config in configs:
            for model in models:
                m = model(**config)
                m.fit(train)
                recommended = m.predict(
                    test, users=sampled_users, items=sampled_items
                )
                ndcg = score(metrics.truncated_ndcg, held_out_items, recommended).mean()
                hit_ratio = score(metrics.hit_ratio, held_out_items, recommended).mean()
                result = dict(
                    name=model.method, trial=i, ndcg=ndcg, hit_ratio=hit_ratio,
                )
                result.update(config)
                results.append(result)

    return results


def _dump_results(results, path):
    df = pd.DataFrame.from_records(results)

    if not os.path.exists(path):
        os.makedirs(os.path.split(path)[0], exist_ok=True)

    df.to_csv(path, index=False)


def ncf(
    input_path: str = "data/movielens-100k/ratings.csv",
    output_path: str = "data/benchmarking/ncf.csv",
    n_trials: int = 3,
    policy: str = "leave_one_out",
    **kwargs: Optional[Any]
):
    df = pd.read_csv(input_path)
    df = df.rename(columns={"userId": "user", "movieId": "item"})

    train_dataset, test_dataset = create_datasets(df, policy=policy, **kwargs)

    _, test_items, _ = test_dataset.to_components(shuffle=False)

    neural_models = [
        neural.GeneralizedMatrixFactorizationModel
    ]

    neural_configs = [
        {"n_factors": 8, "fit_params": {"epochs": 1, "batch_size": 256}},
    ]

    users, items = he_sampling(test_dataset, train_dataset)

    results = _run_trials(
        neural_models,
        neural_configs,
        train_dataset,
        test_dataset,
        users,
        items,
        test_items,
        n_trials=n_trials,
    )

    _dump_results(results, output_path)


def baselines(
    input_path: str = "data/movielens-100k/ratings.csv",
    output_path: str = "data/benchmarking/baselines.csv",
    n_trials: int = 3,
    policy: str = "leave_one_out",
    **kwargs: Optional[Any]
):

    df = pd.read_csv(input_path)
    df = df.rename(columns={"userId": "user", "movieId": "item"})

    train_dataset, test_dataset = create_datasets(df, policy=policy, **kwargs)

    _, test_items, _ = test_dataset.to_components(shuffle=False)

    baseline_models = [
        baseline.AlternatingLeastSquaresModel,
        baseline.BayesianPersonalizedRankingModel,
    ]

    users, items = he_sampling(test_dataset, train_dataset)

    baseline_configs = [
        {"factors": 8, "iterations": 15},
        {"factors": 16, "iterations": 15},
        {"factors": 32, "iterations": 15},
        {"factors": 64, "iterations": 15},
    ]

    results = _run_trials(
        baseline_models,
        baseline_configs,
        train_dataset,
        test_dataset,
        users,
        items,
        test_items,
        n_trials=n_trials,
    )

    _dump_results(results, output_path)


if __name__ == "__main__":
    fire.Fire()
