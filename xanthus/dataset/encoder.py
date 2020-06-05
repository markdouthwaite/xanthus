import numpy as np


class DatasetEncoder:
    def __init__(self, offset=1):
        self._offset = offset
        self.user_mapping = {}
        self.item_mapping = {}
        self.user_feature_mapping = {}
        self.item_feature_mapping = {}

    def fit(self, *args, **kwargs):
        return self.partial_fit(*args, **kwargs)

    def partial_fit(
        self, users=None, items=None, user_features=None, item_features=None
    ):

        if users is not None:
            self.user_mapping = self._fit_mapping(users, self.user_mapping)

        if items is not None:
            self.item_mapping = self._fit_mapping(items, self.item_mapping)

        if user_features is not None:
            self.user_feature_mapping = self._fit_feature_mapping(
                user_features, self.user_feature_mapping, self.user_mapping
            )

        if item_features is not None:
            self.item_feature_mapping = self._fit_feature_mapping(
                item_features, self.item_feature_mapping, self.item_mapping
            )

        return self

    def _fit_feature_mapping(self, data, encodings, aux_encodings):
        encodings = self._fit_mapping(data, encodings, offset=len(aux_encodings),)
        encodings.update(aux_encodings)
        return encodings

    def _fit_mapping(self, data, encodings, offset=0):
        for e in data:
            encodings.setdefault(e, len(encodings) + offset + self._offset)
        return encodings

    def inverse_transform(
        self, users=None, items=None, user_features=None, item_features=None,
    ):
        output = dict()
        if users is not None:
            inv_user_mappings = {v: k for k, v in self.user_mapping.items()}
            output["users"] = np.asarray(list(inv_user_mappings[_] for _ in users))

        if items is not None:
            inv_item_mappings = {v: k for k, v in self.item_mapping.items()}
            output["items"] = np.asarray(list(inv_item_mappings[_] for _ in items))

        if user_features is not None:
            inv_user_feat_mappings = {
                v: k for k, v in self.user_feature_mapping.items()
            }
            output["user_features"] = np.asarray(
                list(inv_user_feat_mappings[_] for _ in user_features)
            )

        if item_features is not None:
            inv_item_feat_mappings = {
                v: k for k, v in self.item_feature_mapping.items()
            }
            output["item_features"] = np.asarray(
                list(inv_item_feat_mappings[_] for _ in item_features)
            )

        return output

    def transform(self, users=None, items=None, user_features=None, item_features=None):
        output = dict()

        if users is not None:
            output["users"] = np.fromiter(
                (self.user_mapping[_] for _ in users), np.int32
            )

        if items is not None:
            output["items"] = np.fromiter(
                (self.item_mapping[_] for _ in items), np.int32
            )

        if user_features is not None:
            output["user_features"] = np.fromiter(
                (self.user_feature_mapping[_] for _ in user_features), np.int32
            )

        if item_features is not None:
            output["item_features"] = np.fromiter(
                (self.item_feature_mapping[_] for _ in item_features), np.int32
            )

        return output

    def fit_transform(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
