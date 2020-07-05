import numpy as np
from xanthus.evaluate import score, metrics

options = np.arange(0, 1000)


actual = []
predicted = []
for i in range(100):
    a = np.random.choice(options, 6)
    p = np.random.choice(a, 3)
    p_ = np.random.choice(options[~np.isin(options, a)], 3)
    actual.append(a)
    predicted.append(p)


print(score(metrics.precision_at_k, actual, predicted).mean())
