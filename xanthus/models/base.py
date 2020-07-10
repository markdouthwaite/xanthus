"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Any, Dict, Callable, Tuple

import numpy as np
from numpy import ndarray

from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam, Optimizer
from tensorflow.keras.losses import BinaryCrossentropy, Loss

from sklearn.model_selection import train_test_split

from xanthus.datasets import Dataset


Metric = Callable[[List[int], List[int], Optional[Any]], float]


class RecommenderModel(ABC):
    """A simple recommender model interface. Inspired by the scikit-learn API."""

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

        Returns
        -------
        output: list
            A list, where each element corresponds to a list of recommendations. If a
            list of 'users' was provided, this will be ordered by this list. If 'users'
            are not provided, it will be ordered by 'dataset.all_users'.

        """


class NeuralRecommenderModel(RecommenderModel):
    """
    RecommenderModels with a Neural flavour, designed to support negative sampling as
    described in [1], and to work with Xanthus Datasets. Built with Keras.

    Parameters
    ----------
    loss: Loss
        A Keras Loss function of your choice. BinaryCrossEntropy is used by default [1].
    optimizer: Optimizer
        A Keras Optimizer of your choice. Adam is used by default [1].
    negative_samples: int
        Specify the number of negative samples (per positive sample) you wish to
        generate during training. Note that negative sampling can dramatically slow
        down your training loop, but can also boost performance in the end.
    fit_params: dict, optional
        Optional fit parameters to be passed to the Keras `fit` call.
    kwargs: any, optional
        Additional keyword arguments.

    Notes
    -----
    * Supports negative sampling. This is currently v. expensive.
    * Subclassing of a Keras Model was considered, but rejected as it limits your
      ability to save a model in a format other than `pickle`.

    References
    ----------
    [1] He et al. https://dl.acm.org/doi/10.1145/3038912.3052569

    See Also
    --------
    tensorflow.ketas.Model.fit


    """

    def __init__(
        self,
        loss: Loss = BinaryCrossentropy(),
        optimizer: Optimizer = Adam(lr=1e-3),
        negative_samples: int = 0,
        n_meta: int = 0,
        fit_params: Optional[Dict[str, Any]] = None,
        **kwargs: Optional[Any],
    ):
        """
        Initialize a NeuralRecommender
        """

        self.model = None
        self._loss = loss
        self._optimizer = optimizer
        self._negative_samples = negative_samples
        self._config = kwargs
        self._n_meta = n_meta
        self._fit_params = fit_params

    def fit(
        self,
        dataset: Dataset,
        **kwargs: Optional[Any],
    ) -> "NeuralRecommenderModel":
        """
        Fit the model to a provided Dataset.

        Parameters
        ----------
        dataset: Dataset
            An input dataset.

        Returns
        -------
        output: NeuralRecommenderModel
            Returns itself.

        See Also
        --------
        xanthus.datasets.Dataset

        """

        n_dim = self._n_meta + 1

        if self.model is None:
            self.model = self._build_model(
                dataset, n_user_dim=n_dim, n_item_dim=n_dim, **self._config
            )
            self.model.compile(optimizer=self._optimizer, loss=self._loss)

        epochs, fit_params = self._unpack_fit_params()

        for i in range(epochs):
            user_x, item_x, y = dataset.to_components(
                negative_samples=self._negative_samples, output_dim=self._n_meta + 1
            )

            tux, vux, tix, vix, ty, vy = train_test_split(
                user_x, item_x, y, test_size=0.1
            )

            self.model.fit(
                [tux, tix],
                ty,
                epochs=i + 1,
                initial_epoch=i,
                **fit_params,
                validation_data=([vux, vix], vy),
                **kwargs,
            )

        return self

    def predict(
        self,
        dataset: Optional[Dataset] = None,
        users: Optional[List[int]] = None,
        items: Optional[List[int]] = None,
        n: int = 3,
        excluded: Optional[List[int]] = None,
    ) -> List[List[int]]:
        """
        Generate predictions (recommendations) from the model. For each provided user,
        the output will be a set of items ranked in order of predicted preference.

        Parameters
        ----------
        dataset: Dataset, optional
            The dataset for which you wish to generate recommendations. If this is
            provided, this object's `all_items` and `users` will be used for the
            purpose of generating recommendations.
        users: list, optional
            An optional array of users for whom you wish to generate recommendations.
        items: list, optional
            An optional array of items you wish to be used in recommendations. This
            may be a subset of items for the purposes of ranking specific subsets of
            items, for example. Maybe you want to see which Star Wars movies are the
            most loved according to the model, for example.
        n: int
            The number of recommendations to be generated per user.
        excluded: list, optional
            An optional array of items to exclude from recommendations.
        Returns
        -------
        output: list
            A list, where each element corresponds to a list of recommendations. If a
            list of 'users' was provided, this will be ordered by this list. If 'users'
            are not provided, it will be ordered by 'dataset.all_users'.

        # Todo:
        # * This needs an overhaul. Some of that will involve re-working the
        #   dataset generators, but this feels v. clunky & extraordinarily slow.

        """

        recommended = []

        users = users if users is not None else dataset.users

        if items is None:
            all_items = dataset.all_items
            items = (all_items for _ in users)

        if self._n_meta > 0:
            items = (dataset.iter_item(e, n_dim=self._n_meta + 1) for e in items)
            users = dataset.iter_user(users, n_dim=self._n_meta + 1)

        for (user, target_items) in zip(users, items):
            # this _list_ unpacks the metadata.
            target_items = np.asarray(list(target_items))

            if len(target_items.shape) == 1:
                target_items = target_items.reshape(-1, 1)

            x = [np.tile(user, len(target_items)), target_items]

            h = self.model(x).numpy().flatten()
            ranked = self._rank(h, n, encodings=target_items[:, 0], excluded=excluded)

            recommended.append(ranked[:n])

        return recommended

    @abstractmethod
    def _build_model(self, dataset: Dataset, **kwargs: Optional[Any]) -> Model:
        """Build a Keras model."""

    def _unpack_fit_params(self, epochs: int = 1) -> Tuple[int, Dict[str, Any]]:
        """Unpack fit parameters, extracting `epochs` for use in training loop."""

        epochs = self._fit_params.get("epochs", epochs)
        fit_params = {k: v for k, v in self._fit_params.items() if k != "epochs"}
        return epochs, fit_params

    @staticmethod
    def _rank(
        w: ndarray,
        n: int,
        encodings: Optional[ndarray] = None,
        excluded: Optional[ndarray] = None,
    ) -> ndarray:
        """
        A utility for simply ranking predictions.

        Parameters
        ----------
        w: ndarray
            The input weights/scores to rank.
        n: int
            The total number of recommendations you require.
        encodings: ndarray, optional
            Optional encodings that should map to the elements in the `w` array.
        excluded: ndarray, optional
            Optional elements that should be excluded from the output rankings.

        Returns
        -------
        output: ndarray
            An array at most `n` elements in length.

        """

        ranked = w.argsort()[::-1]

        if encodings is not None:
            ranked = encodings[ranked]

        if excluded is not None:
            ranked = ranked[~np.isin(ranked, excluded)]

        return ranked[:n]
