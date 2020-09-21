import uuid
import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd


TIMESTAMP_FORMAT = "%H:%M:%S.%f %y-%m-%d"


def benchmark(manager, epochs, **kwargs):
    logger = logging.getLogger(f"Benchmark ({manager.name}|{kwargs})")
    start = datetime.now()
    records = []
    for epoch in range(epochs):
        logger.info(f"Running epoch {epoch + 1} of {epochs}...")
        manager.update(1)
        metrics = manager.metrics(**kwargs)
        metrics["epoch"] = epoch + 1
        records.append(metrics)

    end = datetime.now()
    info = dict(
        start=start.strftime(TIMESTAMP_FORMAT),
        end=end.strftime(TIMESTAMP_FORMAT),
        elapsed=(end - start).seconds,
        params=manager.params(),
    )

    return records, info


def save(experiment, manager, records, info=None, root=None, identifier=None):

    identifier = identifier or uuid.uuid4().hex[:6]

    if root is not None:
        path = Path(root) / experiment / manager.name / identifier
    else:
        path = Path(experiment) / manager.name / identifier

    path.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame.from_records(records)
    df.to_csv(path / "results.csv")

    if info is not None:
        json.dump(info, (path / "info.json").open("w"))
