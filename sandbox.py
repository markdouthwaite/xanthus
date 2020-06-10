import time
import uuid
import numpy as np
import pandas as pd

from xanthus.evaluate.utils import split

np.random.seed(42)

n_users = 5000000
n_items = 10000
n_trans = 10000000
users = np.asarray([uuid.uuid4().hex for _ in range(n_users)])
items = np.asarray([uuid.uuid4().hex for _ in range(n_items)])

trans = np.c_[
    np.random.choice(users, size=n_trans),
    np.random.choice(items, size=n_trans),
]

df = pd.DataFrame(data=trans, columns=["user", "item"])
print(df.shape)
df = df.drop_duplicates()
print(df.shape)
print("Done...")
t1 = time.time()
split(df, frac_train=0.75, n_test=1)
t2 = time.time()

print(t2 - t1)
