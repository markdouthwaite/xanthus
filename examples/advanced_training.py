"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

import numpy as np
import pandas as pd

from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from sklearn.model_selection import train_test_split

from xanthus import datasets, models
from xanthus.evaluate import create_rankings, score, metrics

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
# remember: you can add custom callbacks too!

# and now to the old-school model training bit.
datasets.movielens.download("ml-latest-small")

df = pd.read_csv("data/ml-latest-small/ratings.csv")
df = df.rename(columns={"userId": "user", "movieId": "item"})

# for expedience, you can use `datasets.build` to do a lot of the setup for you
# - at the loss of flexibility and transparency, of course.
train_dataset, test_dataset = datasets.build(df, policy="leave_one_out")

# prepare model
model = models.GeneralizedMatrixFactorization(
    n=train_dataset.user_dim, m=train_dataset.item_dim
)
model.compile(optimizer=Adam(), loss=BinaryCrossentropy())

# get training data
user_x, item_x, y = train_dataset.to_components(
    negative_samples=1, aux_matrix=test_dataset.interactions
)
train_user_x, val_user_x, train_item_x, val_item_x, train_y, val_y = train_test_split(
    user_x, item_x, y, test_size=0.2
)

model.fit(
    [train_user_x, train_item_x],
    train_y,
    epochs=1,
    batch_size=256,
    validation_data=([val_user_x, val_item_x], val_y),
    callbacks=[tensorboard, early_stop],
)

# get evaluation data
users, items = create_rankings(test_dataset, train_dataset, n_samples=100, unravel=True)
_, test_items, _ = test_dataset.to_components(shuffle=False)

# generate scores and evaluate
scores = model.predict([users, items], verbose=1)
recommended = models.utils.reshape_recommended(
    users.reshape(-1, 1), items.reshape(-1, 1), scores, 10, mode="array"
)

print("nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.hit_ratio, test_items, recommended).mean())
