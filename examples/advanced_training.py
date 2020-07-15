"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import numpy as np
import pandas as pd

from tensorflow.keras import callbacks

from xanthus.models import neural
from xanthus.evaluate import he_sampling, score, metrics
from xanthus.utils import create_datasets

np.random.seed(42)

# setup your Tensorboard callback. This will write logs to the `/logs` directory.
# you can start Tensorboard while your model is training by running:
# tensorboard --logdir=examples/logs
# Note - you may need to install Tensorboard first!
tensorboard = callbacks.TensorBoard(log_dir="./logs", profile_batch=5)

# setup your Early Stopping protocol. This will terminate your training early if the
# validation loss for your model does not improve after a given period of time.
early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=1e-4,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)
# you can add custom callbacks too!

# and now to the old-school model training bit.
df = pd.read_csv("../data/movielens-100k/ratings.csv")
df = df.rename(columns={"userId": "user", "movieId": "item"})

# for expedience, you can use `create_datasets` to do a lot of the setup for you
# - at the loss of flexibility and transparency, of course.
train_dataset, test_dataset = create_datasets(df, policy="leave_one_out")

users, items = he_sampling(test_dataset, train_dataset)
_, test_items, _ = test_dataset.to_components(shuffle=False)

model = neural.GeneralizedMatrixFactorizationModel(
    fit_params=dict(epochs=50, batch_size=256),
    n_factors=8,
    negative_samples=4,
)

# make sure to pass your 'callbacks' arguments in here as a list!
model.fit(train_dataset, callbacks=[tensorboard, early_stop])

recommended = model.predict(test_dataset, users=users, items=items, n=10)

print("t-nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.hit_ratio, test_items, recommended).mean())
