"""
The MIT License

Copyright (c) 2018-2020 Mark Douthwaite
"""

from typing import Optional, Any, List, Dict

import numpy as np
from numpy import ndarray


class DatasetEncoder:
    def __init__(self, offset: int = 1) -> None:
        self._offset = offset
        self.user_mapping: Dict[str, int] = {}
        self.item_mapping: Dict[str, int] = {}
        self.user_tag_mapping: Dict[str, int] = {}
        self.item_tag_mapping: Dict[str, int] = {}

    def fit(self, *args: Optional[Any], **kwargs: Optional[Any]) -> "DatasetEncoder":
        return self.partial_fit(*args, **kwargs)

    def transform(
        self,
        users: Optional[List[str]] = None,
        items: Optional[List[str]] = None,
        user_tags: Optional[List[str]] = None,
        item_tags: Optional[List[str]] = None,
    ) -> Dict[str, ndarray]:

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
            output["user_features"] = np.fromiter(
                (self.user_tag_mapping[_] for _ in user_tags), np.int32
            )

        if item_tags is not None:
            output["item_features"] = np.fromiter(
                (self.item_tag_mapping[_] for _ in item_tags), np.int32
            )

        return output

    def fit_transform(
        self, *args: Optional[Any], **kwargs: Optional[Any]
    ) -> Dict[str, ndarray]:

        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)

    def partial_fit(
        self,
        users: Optional[List[str]] = None,
        items: Optional[List[str]] = None,
        user_features: Optional[List[str]] = None,
        item_features: Optional[List[str]] = None,
    ) -> "DatasetEncoder":

        if users is not None:
            self.user_mapping = self._fit_mapping(users, self.user_mapping)

        if items is not None:
            self.item_mapping = self._fit_mapping(items, self.item_mapping)

        if user_features is not None:
            self.user_tag_mapping = self._fit_feature_mapping(
                user_features, self.user_tag_mapping, self.user_mapping
            )

        if item_features is not None:
            self.item_tag_mapping = self._fit_feature_mapping(
                item_features, self.item_tag_mapping, self.item_mapping
            )

        return self

    def inverse_transform(
        self,
        users: Optional[List[int]] = None,
        items: Optional[List[int]] = None,
        user_features: Optional[List[int]] = None,
        item_features: Optional[List[int]] = None,
    ) -> Dict[str, ndarray]:

        output = dict()

        if users is not None:
            inv_user_mappings = {v: k for k, v in self.user_mapping.items()}
            output["users"] = np.asarray(list(inv_user_mappings[_] for _ in users))

        if items is not None:
            inv_item_mappings = {v: k for k, v in self.item_mapping.items()}
            output["items"] = np.asarray(list(inv_item_mappings[_] for _ in items))

        if user_features is not None:
            inv_user_feat_mappings = {v: k for k, v in self.user_tag_mapping.items()}
            output["user_features"] = np.asarray(
                list(inv_user_feat_mappings[_] for _ in user_features)
            )

        if item_features is not None:
            inv_item_feat_mappings = {v: k for k, v in self.item_tag_mapping.items()}
            output["item_features"] = np.asarray(
                list(inv_item_feat_mappings[_] for _ in item_features)
            )

        return output

    def get_params(self) -> Dict[str, Dict[str, int]]:
        return dict(
            user_mapping=self.user_mapping,
            item_mapping=self.item_mapping,
            user_tag_mapping=self.user_tag_mapping,
            item_tag_mapping=self.item_tag_mapping,
        )

    def set_params(self, params: Dict[str, Dict[str, int]]) -> "DatasetEncoder":
        self.user_mapping = params.get("user_mapping", self.user_mapping)
        self.item_mapping = params.get("item_mapping", self.item_mapping)
        self.user_tag_mapping = params.get("user_tag_mapping", self.user_tag_mapping)
        self.item_tag_mapping = params.get("item_tag_mapping", self.item_tag_mapping)
        return self

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
