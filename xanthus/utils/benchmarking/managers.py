from typing import Any, Dict, Tuple
import numpy as np

from xanthus.evaluate import metrics, create_rankings, score
from xanthus.models.baseline import MatrixFactorization as MFModel


class ModelManager:
    def __init__(self, name: str, datasets: Tuple, **params: Any) -> None:
        self.name = name
        self._params = params
        self.train = datasets[0]
        self.val = datasets[1]

    def update(self, epochs: int) -> None:
        pass

    def metrics(self, **kwargs) -> Dict[str, float]:
        pass

    def params(self) -> Dict[str, Any]:
        pass


class BaselineModelManager(ModelManager):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._model = MFModel(self.name, **self._params, iterations=1)

    def update(self, epochs: int) -> None:
        self._model.fit(self.train)

    def metrics(self, min_k: int = 1, max_k: int = 10) -> Dict[str, float]:

        users, items, _ = self.val.to_components(shuffle=False)

        _, ranked_items = create_rankings(self.val, self.train)

        recommended = self._model.predict(
            self.val, users=users, items=ranked_items, n=10
        )

        pak = {}
        ndcg = {}
        for k in range(min_k, max_k):
            pak[f"hr{k}"] = score(metrics.pak, items, recommended).mean()
            ndcg[f"ndcg{k}"] = score(metrics.truncated_ndcg, items, recommended).mean()

        return dict(**pak, **ndcg)


class NeuralModelManager(ModelManager):
    def metrics(self, min_k: int = 1, max_k: int = 10) -> Dict[str, float]:
        return dict(
            **{f"ndcg{_}": np.random.rand() for _ in range(min_k, max_k + 1)},
            **{f"hr{_}": np.random.rand() for _ in range(min_k, max_k + 1)},
            loss=float(np.random.rand()),
        )
