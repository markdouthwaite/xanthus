from typing import Dict

import numpy as np

from sklearn.model_selection import train_test_split

from xanthus.evaluate import metrics, create_rankings, score
from xanthus.models.baseline import MatrixFactorization as MFModel
from xanthus.models import (
    MultiLayerPerceptron,
    GeneralizedMatrixFactorization,
    NeuralMatrixFactorization,
    utils,
)

from xanthus.utils.benchmarking import ModelManager


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

        recommended = np.asarray(recommended)

        pak = {}
        ndcg = {}

        for k in range(min_k, max_k + 1):
            pak[f"hr{k}"] = score(metrics.pak, items, recommended[:, :k]).mean()
            ndcg[f"ndcg{k}"] = score(
                metrics.truncated_ndcg, items, recommended[:, :k]
            ).mean()

        return dict(**pak, **ndcg)


class NeuralModelManager(ModelManager):

    _models = {
        "gmf": GeneralizedMatrixFactorization,
        "nmf": NeuralMatrixFactorization,
        "mlp": MultiLayerPerceptron,
    }

    def __init__(
        self,
        *args,
        optimizer="adam",
        loss="binary_crossentropy",
        samples=3,
        batch_size=256,
        factors=8,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.samples = samples
        self.batch_size = batch_size
        self._model = self.get_model(factors, optimizer, loss)

        self.users, self.items = create_rankings(
            self.val, self.train, n_samples=100, unravel=True
        )

        _, test_items, _ = self.val.to_components(shuffle=False)
        self.test_items = test_items

    def get_model(self, factors, optimizer, loss):

        model_class = self._models[self.name]

        if self.name == "mlp":
            model = model_class(
                n=self.train.user_dim,
                m=self.train.item_dim,
                layers=self.get_layers(factors),
            )
        elif self.name == "nmf":
            model = model_class(
                self.train.user_dim,
                m=self.train.item_dim,
                factors=factors,
                layers=self.get_layers(factors),
            )
        else:
            model = model_class(
                n=self.train.user_dim, m=self.train.item_dim, factors=factors
            )

        model.compile(optimizer=optimizer, loss=loss)

        return model

    def get_layers(self, factors, n=3):
        layers = [factors]

        for i in range(2, n + 1):
            layers.append(layers[-1] * 2)

        return tuple(layers[::-1])

    def update(self, epochs: int) -> None:
        user_x, item_x, y = self.train.to_components(
            negative_samples=self.samples, aux_matrix=self.val.interactions
        )
        (
            train_user_x,
            val_user_x,
            train_item_x,
            val_item_x,
            train_y,
            val_y,
        ) = train_test_split(user_x, item_x, y, test_size=0.2)

        self._model.fit(
            [train_user_x, train_item_x],
            train_y,
            epochs=1,
            batch_size=self.batch_size,
            validation_data=([val_user_x, val_item_x], val_y),
        )

    def metrics(self, min_k: int = 1, max_k: int = 10) -> Dict[str, float]:
        scores = self._model.predict([self.users, self.items])

        recommended = utils.reshape_recommended(
            self.users.reshape(-1, 1),
            self.items.reshape(-1, 1),
            scores,
            max_k,
            mode="array",
        )

        pak = {}
        ndcg = {}

        for k in range(min_k, max_k + 1):
            pak[f"hr{k}"] = score(
                metrics.pak, self.test_items, recommended[:, :k]
            ).mean()
            ndcg[f"ndcg{k}"] = score(
                metrics.truncated_ndcg, self.test_items, recommended[:, :k]
            ).mean()

        return dict(**pak, **ndcg)
