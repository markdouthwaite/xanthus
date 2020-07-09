import numpy as np
import pandas as pd
from tensorflow.keras import callbacks
from xanthus.datasets import Dataset, DatasetEncoder, utils
from xanthus.models import MultiLayerPerceptronModel as GMFModel, baseline
from xanthus.evaluate import he_sampling, score, metrics, leave_one_out

np.random.seed(42)

# df = pd.read_csv(
#     "../data/movielens-1m/ratings.dat",
#     names=["userId", "movieId", "rating", "timestamp"],
#     delimiter="::",
# )

df = pd.read_csv("data/movielens-100k/ratings.csv")

raw_meta = pd.read_csv("data/movielens-100k/movies.csv")
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

model = GMFModel(
    fit_params=dict(epochs=5, batch_size=256), n_factors=16, negative_samples=4, n_meta=2
)
# model = baseline.MatrixFactorizationModel(method="als", factors=16, iterations=15)

model.fit(train_dataset)

users, items = he_sampling(test_dataset, train_dataset)

recommended = model.predict(test_dataset, users=users, items=items, n=10)

print("t-nDCG", score(metrics.truncated_ndcg, test_items, recommended).mean())
print("HR@k", score(metrics.hit_ratio, test_items, recommended).mean())

"""
# GMF - 100k @ 15 (n=64)
t-nDCG 0.5882448726571842
HR@k 0.8114754098360656

# ALS - 100k @ 15 (n=64)
t-nDCG 0.6088589090862165
HR@k 0.7819672131147541

# GMF - 100k @ 5 (n=8, m=0)
t-nDCG 0.47441618099927274
HR@k 0.7344262295081967

# GMF - 100k @ 5 (n=8, m=1)
t-nDCG 0.46211201152582154
HR@k 0.7262295081967213

# GMF - 100k @ 15 (n=8, m=1)
t-nDCG 0.5169057877307099
HR@k 0.7836065573770492

# GMF - 100k @ 15 (n=8, m=0)
t-nDCG 0.5056338707603265
HR@k 0.7754098360655738

# GMF - 100k @ 15 (n=16, m=0)
t-nDCG 0.5481534922808121
HR@k 0.8180327868852459

# GMF - 100k @ 15 (n=16, m=1)
t-nDCG 0.5438347316079841
HR@k 0.8

# GMF - 100k @ 15 (n=16, m=2, s=1)
t-nDCG 0.5674754101237157
HR@k 0.8245901639344262

# GMF - 100k @ 15 (n=16, m=3, s=4)
t-nDCG 0.6081511034467256
HR@k 0.8295081967213115

# ALS - 100k @ 15 (n=16)
t-nDCG 0.6029315694478758
HR@k 0.8180327868852459

"""


tensorboard = callbacks.TensorBoard(log_dir='./logs', profile_batch=5)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto',
    baseline=None, restore_best_weights=False
)