"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Any, List, Dict, Callable

import numpy as np
from numpy import ndarray
from pandas import DataFrame

# test: different offsets.


class DatasetEncoder:
    """
    A class for simply encoding and decoding recommendation datasets in the spirit of
    a Scikit-Learn Transformer (but not _actually_ a transformer!).

    The objective of this class is to help you manage, save and load your encodings, and
    to give you a few little utilities to help make building a recommendation model that
    bit simpler.

    Parameters
    ----------
    offset: int
        The point at which encodings 'start'. By default, '0' is reserved (hence '1').

    """

    def __init__(self, offset: int = 1) -> None:
        """
        Initialize the DatasetEncoder.
        """

        self._offset = offset
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.user_tag_mapping: Dict[str, int] = {}
        self.item_tag_mapping: Dict[str, int] = {}

    def fit(self, *args: Optional[Any], **kwargs: Optional[Any]) -> "DatasetEncoder":
        """Fit the DatasetEncoder."""

        return self.partial_fit(*args, **kwargs)

    def transform(
        self,
        users: Optional[List[str]] = None,
        items: Optional[List[str]] = None,
        user_tags: Optional[List[str]] = None,
        item_tags: Optional[List[str]] = None,
    ) -> Dict[str, ndarray]:
        """

        Parameters
        ----------
        users
        items
        user_tags
        item_tags

        Returns
        -------

        """

        output = dict()

        if users is not None:
            output["users"] = np.fromiter(
                (self.user_mapping[_] for _ in users), np.int32
            )

        if items is not None:
            output["items"] = np.fromiter(
                (self.item_mapping[_] for _ in items), np.int32
            )

        if user_tags is not None:
            output["user_tags"] = np.fromiter(
                (self.user_tag_mapping[_] for _ in user_tags), np.int32
            )

        if item_tags is not None:
            output["item_tags"] = np.fromiter(
                (self.item_tag_mapping[_] for _ in item_tags), np.int32
            )

        return output

    def fit_transform(
        self, *args: Optional[Any], **kwargs: Optional[Any]
    ) -> Dict[str, ndarray]:
        """Fit the DatasetEncoder, then transform it. Simples."""

        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def partial_fit(
        self,
        users: Optional[List[str]] = None,
        items: Optional[List[str]] = None,
        user_tags: Optional[List[str]] = None,
        item_tags: Optional[List[str]] = None,
    ) -> "DatasetEncoder":
        """
        Fit the DatasetEncoder.

        This will update encodings to include any new user/item values.

        Parameters
        ----------
        users: list, optional
            A list of users to encode.
        items: list, optional
            A list of items to encode.
        user_tags: list, optional
            A list of user features to encode.
        item_tags: list, optional
            A list of item features to encode.

        Returns
        -------
        output: DatasetEncoder
            A newly updated DatasetEncoder.

        """

        if users is not None:
            self.user_mapping = self._fit_mapping(users, self.user_mapping)

        if items is not None:
            self.item_mapping = self._fit_mapping(items, self.item_mapping)

        if user_tags is not None:
            self.user_tag_mapping = self._fit_feature_mapping(
                user_tags, self.user_tag_mapping, self.user_mapping
            )

        if item_tags is not None:
            self.item_tag_mapping = self._fit_feature_mapping(
                item_tags, self.item_tag_mapping, self.item_mapping
            )

        return self

    def inverse_transform(
        self,
        users: Optional[List[int]] = None,
        items: Optional[List[int]] = None,
        user_tags: Optional[List[int]] = None,
        item_tags: Optional[List[int]] = None,
    ) -> Dict[str, ndarray]:
        """

        Parameters
        ----------
        users
        items
        user_tags
        item_tags

        Returns
        -------

        """

        output = dict()

        if users is not None:
            inv_user_mappings = {v: k for k, v in self.user_mapping.items()}
            output["users"] = np.asarray(list(inv_user_mappings[_] for _ in users))

        if items is not None:
            inv_item_mappings = {v: k for k, v in self.item_mapping.items()}
            output["items"] = np.asarray(list(inv_item_mappings[_] for _ in items))

        if user_tags is not None:
            inv_user_feat_mappings = {v: k for k, v in self.user_tag_mapping.items()}
            output["user_tags"] = np.asarray(
                list(inv_user_feat_mappings[_] for _ in user_tags)
            )

        if item_tags is not None:
            inv_item_feat_mappings = {v: k for k, v in self.item_tag_mapping.items()}
            output["item_tags"] = np.asarray(
                list(inv_item_feat_mappings[_] for _ in item_tags)
            )

        return output

    def get_params(self) -> Dict[str, Dict[str, int]]:
        """

        Returns
        -------

        """
        return dict(
            user_mapping=self.user_mapping,
            item_mapping=self.item_mapping,
            user_tag_mapping=self.user_tag_mapping,
            item_tag_mapping=self.item_tag_mapping,
        )

    def set_params(self, params: Dict[str, Dict[str, int]]) -> "DatasetEncoder":
        """

        Parameters
        ----------
        params

        Returns
        -------

        """
        self.user_mapping = params.get("user_mapping", self.user_mapping)
        self.item_mapping = params.get("item_mapping", self.item_mapping)
        self.user_tag_mapping = params.get("user_tag_mapping", self.user_tag_mapping)
        self.item_tag_mapping = params.get("item_tag_mapping", self.item_tag_mapping)
        return self

    def to_df(
        self,
        users: List[int],
        items: List[List[int]],
        user_col: str = "user",
        item_cols: str = "item_{0}",
        transform: Callable[[List[int]], List[str]] = None,
    ) -> DataFrame:
        """

        Parameters
        ----------
        users
        items
        user_col
        item_cols
        transform

        Returns
        -------

        """

        if len(users) != len(items):
            raise ValueError(
                f"The total number of users ('{len(users)}') does not match the total "
                f"number of item recommendation rows ('{len(items)}')."
            )

        # create column headers.
        n = len(items[0])
        transform = transform or self.inverse_transform
        columns = [user_col, *(item_cols.format(i) for i in range(n))]

        # inverse-transform user/item arrays.
        users = transform(users=users)["users"]
        items = [transform(items=items[i])["items"] for i in range(len(items))]

        # stack users and items.
        data = np.c_[np.asarray(users), np.asarray(items)]

        return DataFrame(data=data, columns=columns)

    def _fit_feature_mapping(
        self, x: List[str], encodings: Dict[str, int], aux_encodings: Dict[str, int]
    ) -> Dict[str, int]:
        encodings = self._fit_mapping(x, encodings, offset=len(aux_encodings),)
        encodings.update(aux_encodings)
        return encodings

    def _fit_mapping(
        self, x: List[str], encodings: Dict[str, int], offset: int = 0
    ) -> Dict[str, int]:
        for e in x:
            encodings.setdefault(e, len(encodings) + offset + self._offset)
        return encodings
