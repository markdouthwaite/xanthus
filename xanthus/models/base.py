from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Callable

import numpy as np

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.losses import BinaryCrossentropy, Loss

from xanthus.datasets import Dataset


Metric = Callable[[List[int], List[int], Optional[Any]], float]


class RecommenderModel(ABC):
    """A simple recommender model interface."""

    @abstractmethod
    def fit(self, dataset: Dataset) -> "RecommenderModel":
        """Fit the model on the provided dataset."""

    @abstractmethod
    def predict(
        self,
        dataset: Dataset,
        users: Optional[List[int]] = None,
        items: Optional[List[int]] = None,
        n: int = 3,
        excluded: Optional[List[int]] = None,
        **kwargs: Optional[Any],
    ) -> List[List[int]]:
        """
        Generate 'n' predictions for the given dataset (and optionally, provided users
        and items).

        Parameters
        ----------
        dataset: Dataset
            The dataset to be used for generating recommendations.
        users: list, optional
            An optional list of users for whom you wish to generate recommendations.
        items: list, optional
            An optional list of items for which you wish to generate recommendations.
            This should map to individual users (either those in 'users' or in
            dataset.all_users).
        n: int
            The total number of recommendations for each user.
        excluded: list, optional
            An optional list of items to exclude from recommendations.
        kwargs: any, optional
            Additional predict params.

        Returns
        -------
        output: list
            A list, where each element corresponds to a list of recommendations. If a
            list of 'users' was provided, this will be ordered by this list. If 'users'
            are not provided, it will be ordered by 'dataset.all_users'.

        """

    # @abstractmethod
    # def save(self, filepath: str) -> None:
    #     """"""
    #
    # @abstractmethod
    # def load(self, filepath: str) -> None:
    #     """"""


class NeuralRecommenderModel(RecommenderModel):
    """

    Notes
    -----
    * Supports negative sampling. This is currently v. expensive.
    * Does not currently support partial/online training.
    """

    def __init__(
        self,
        loss: Loss = BinaryCrossentropy(),
        optimizer: Optimizer = Adam(lr=1e-3),
        negative_samples: int = 0,
        fit_params: Optional[Dict[str, Any]] = None,
        metrics: Optional[List[Metric]] = None,
        **kwargs: Optional[Any],
    ):
        self.model = None
        self._loss = loss
        self._optimizer = optimizer
        self._negative_samples = negative_samples
        self._config = kwargs
        self._metrics = metrics
        self._fit_params = fit_params

    @abstractmethod
    def _build_model(self, dataset: Dataset, **kwargs: Optional[Any]) -> Model:
        """"""

    @staticmethod
    def _rank(w, n, encodings=None, excluded=None):
        ranked = w.argsort()[::-1]

        if encodings is not None:
            ranked = encodings[ranked]

        if excluded is not None:
            ranked = ranked[~np.isin(ranked, excluded)]

        return ranked[:n]

    def fit(self, dataset: Dataset) -> "NeuralRecommenderModel":
        model = self._build_model(dataset, **self._config)
        model.compile(optimizer=self._optimizer, loss=self._loss)

        if "epochs" in self._fit_params:
            epochs = self._fit_params["epochs"]
            fit_params = {k: v for k, v in self._fit_params.items() if k != "epochs"}
        else:
            epochs = 1
            fit_params = self._fit_params

        for i in range(epochs):
            user_x, item_x, y = dataset.to_arrays(
                negative_samples=self._negative_samples
            )

            model.fit(
                [user_x, item_x], y, epochs=i + 1, initial_epoch=i, **fit_params,
            )

        self.model = model

        return self

    def predict(
        self,
        dataset: Dataset,
        users: Optional[List[int]] = None,
        items: Optional[List[int]] = None,
        n: int = 3,
        excluded: Optional[List[int]] = None,
        **kwargs: Optional[Any],
    ) -> List[List[int]]:

        recommended = []
        items = items if items is not None else dataset.all_items
        users = users if users is not None else dataset.users

        for i, user in enumerate(users):
            x = [np.asarray([user] * len(items[i])), items[i]]
            h = self.model(x).numpy().flatten()
            ranked = self._rank(h, n, encodings=items[i], excluded=excluded)
            recommended.append(ranked[:n])

        return recommended
