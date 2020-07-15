import numpy as np
import pandas as pd
from tensorflow.keras import callbacks
from xanthus.datasets import Dataset, DatasetEncoder, utils
from xanthus.models import neural, baseline
from xanthus.evaluate import he_sampling, score, metrics, leave_one_out, split
from xanthus.utils import create_datasets

np.random.seed(42)


tensorboard = callbacks.TensorBoard(log_dir='./logs', profile_batch=5)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=True
)


df = pd.read_csv("../data/movielens-100k/ratings.csv")

raw_meta = pd.read_csv("../data/movielens-100k/movies.csv")
raw_meta = raw_meta.rename(columns={"movieId": "item"})

item_meta = utils.fold(raw_meta, "item", ["genres"],
                       fn=lambda s: (t.lower() for t in s.split("|")))

df = df.rename(columns={"userId": "user", "movieId": "item"})

train, test = leave_one_out(df)

encoder = DatasetEncoder()
encoder.fit(df["user"], df["item"].append(item_meta["item"]), item_tags=item_meta["tag"])

train_dataset = Dataset.from_df(
    train, encoder=encoder, normalize=utils.as_implicit, item_meta=item_meta
)

items = train_dataset.iter_item(df["item"], n_dim=3)

test_dataset = Dataset.from_df(
    test, encoder=encoder, normalize=utils.as_implicit, item_meta=item_meta
)

_, test_items, _ = test_dataset.to_components(shuffle=False)

model = neural.GeneralizedMatrixFactorizationModel(
    fit_params=dict(epochs=1, batch_size=256), n_factors=8, negative_samples=4, n_meta=2
)

model.fit(train_dataset, callbacks=[tensorboard, early_stop])

users, items = he_sampling(test_dataset, train_dataset)

recommended = model.predict(test_dataset, users=users, items=items, n=10)

print("t-nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.hit_ratio, test_items, recommended).mean())
