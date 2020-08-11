"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Any, List, Dict, Callable

import numpy as np
from numpy import ndarray
from pandas import DataFrame


class DatasetEncoder:
    """
    A class for simply encoding and decoding recommendation datasets in the spirit of
    a Scikit-Learn Transformer (but not _actually_ a transformer!).

    The objective of this class is to help you manage, save and load your encodings, and
    to give you a few little utilities to help make building a recommendation model that
    bit simpler.

    Examples
    --------

    >>> from xanthus.datasets import DatasetEncoder
    >>> users = ["jane.smith@email.com", "john.appleseed@email.com"]
    >>> items = ["action-movie-0001", "action-movie-0002"]
    >>> encoder = DatasetEncoder()
    >>> encoded = encoder.fit_transform(users=users, items=items)
    >>> output = encoder.to_df(encoded["users"],
    ...                        [encoded["items"] for _ in encoded["users"]])
    >>> print(output)
                             id             item_0             item_1
    0      jane.smith@email.com  action-movie-0001  action-movie-0002
    1  john.appleseed@email.com  action-movie-0001  action-movie-0002

    """

    def __init__(self) -> None:
        """
        Initialize the DatasetEncoder.
        """

        self._offset = 1
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
        Transform a given set of users, items or their related metadata into a set of
        encodings.

        Parameters
        ----------
        users: list
            A list of user IDs to transform into encodings.
        items: list
            A list of item IDs to transform into encodings.
        user_tags: list
            A list of item meta data to transform into encodings.
        item_tags: list
            A list of item meta data to transform into encodings.

        Returns
        -------
        output: dict
            A dictionary where each input (users, items, user_tags, item_tags)
            that has been passed to the method is returned as a key mapped to an array
            of encoded values. The order of input values is preserved.

        Notes
        -----
        * The elements in users/items and user_tags/item_tags need not be aligned for
          this method to work (i.e. the list of users and items need not be valid
          user item pairs in the input set). Indeed, you can encode only users, only
          items, their metadata or any combination of the set.

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
        Given a set of encoded inputs, return their original values.

        Parameters
        ----------
        users: list, optional
            A list of users to decode.
        items: list, optional
            A list of items to decode.
        user_tags: list, optional
            A list of user features to decode.
        item_tags: list, optional
            A list of item features to decode.

        Returns
        -------
        output: dict
            A dictionary where each encoded input (users, items, user_tags, item_tags)
            that has been passed to the method is returned as a key mapped to an array
            of decoded values. The order of input values is preserved.

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
        """Get the encodings of the current DatasetEncoder."""

        return dict(
            user_mapping=self.user_mapping,
            item_mapping=self.item_mapping,
            user_tag_mapping=self.user_tag_mapping,
            item_tag_mapping=self.item_tag_mapping,
        )

    def set_params(self, params: Dict[str, Dict[str, int]]) -> "DatasetEncoder":
        """Set the encodings to one of your own. What could possibly go wrong?!"""

        self.user_mapping = params.get("user_mapping", self.user_mapping)
        self.item_mapping = params.get("item_mapping", self.item_mapping)
        self.user_tag_mapping = params.get("user_tag_mapping", self.user_tag_mapping)
        self.item_tag_mapping = params.get("item_tag_mapping", self.item_tag_mapping)
        return self

    def to_df(
        self,
        targets: List[int],
        items: List[List[int]],
        mode: str = "users",
        target_col: str = "id",
        item_col_fmt: str = "item_{0}",
        transform: Callable[[List[int]], List[str]] = None,
    ) -> DataFrame:
        """
        Transform a list of users and associated items into a Pandas DataFrame.

        This method is intended to act as a utility for converting *encoded*
        user-recommendation sets into DataFrame format for subsequent post-processing
        and analysis. It is also intended to support item-item 'recommendations' too
        (hence use of 'mode', and 'targets').

        Parameters
        ----------
        targets: list
            A list of encoded targets (e.g. users).
        items: list
            A list, where each element is a list of encoded items mapped to a given
            user (i.e. the user at {i} corresponds to the set of items at {i} in this
            list.
        mode: str
            The mode to use when creating the output frame. Either 'users' or 'items'.
            In the former case, 'targets' will be treated as users, in the latter, as
            items.
        target_col: str
            The name of the output column associated with decoded targets.
        item_col_fmt: str
            The format of the field names associated with items.
        transform: callable, optional
            An optional callable object to be used to transform encodings. If not
            provided, the `inverse_transform` method will be used. This can be
            overridden to postprocess decoded items (e.g. to map to SKUs or descriptive
            names).

        Returns
        -------
        output: DataFrame
            A DataFrame containing _decoded_ users and items associated with them.

        Notes
        -----
        * The item list should correspond to items associated with each of the
          users in the input list. For example, this could be one or more items
          recommended to that users.

        """

        if len(targets) != len(items):
            raise ValueError(
                f"The total number of users ('{len(targets)}') does not match the "
                f"total number of item recommendation rows ('{len(items)}')."
            )

        # create column headers.
        n = len(items[0])
        transform = transform or self.inverse_transform
        columns = [target_col, *(item_col_fmt.format(i) for i in range(n))]

        # inverse-transform user/item arrays.
        if mode == "users":
            targets = transform(users=targets)["users"]
        else:
            targets = transform(items=targets)["items"]

        items = [transform(items=items[i])["items"] for i in range(len(items))]

        # stack users and items.
        data = np.c_[np.asarray(targets), np.asarray(items)]

        return DataFrame(data=data, columns=columns)

    def _fit_feature_mapping(
        self, x: List[str], encodings: Dict[str, int], aux_encodings: Dict[str, int]
    ) -> Dict[str, int]:
        """Generate feature encodings for a given decoded list of features 'x'."""

        encodings = self._fit_mapping(x, encodings, offset=len(aux_encodings),)
        encodings.update(aux_encodings)
        return encodings

    def _fit_mapping(
        self, x: List[str], encodings: Dict[str, int], offset: int = 0
    ) -> Dict[str, int]:
        """Generate encodings for a given decoded list 'x'."""

        for e in x:
            encodings.setdefault(e, len(encodings) + offset + self._offset)
        return encodings
