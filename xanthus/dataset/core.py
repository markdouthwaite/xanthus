import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, users, items, ratings, user_features=None, item_features=None):
        self._users = users
        self._items = items
        self._ratings = self._normalise_ratings(ratings)

        self._user_features = user_features
        self._item_features = item_features

        self._user_encodings = {k: i for i, k in enumerate(np.unique(users))}
        self._item_encodings = {k: i for i, k in enumerate(np.unique(items))}

        if self._user_features is not None:
            n_users = len(self._user_features)
            self._user_encodings = {
                k: i + n_users for i, k in enumerate(np.unique(user_features))
            }

        if self._item_features is not None:
            n_items = len(self._item_features)
            self._item_encodings = {
                k: i + n_items for i, k in enumerate(np.unique(item_features))
            }

    @staticmethod
    def _normalise_ratings(ratings):
        ratings = (ratings - np.min(ratings)) / (np.max(ratings) - np.min(ratings))
        return ratings

    @property
    def encoded_users(self):
        for i, user in enumerate(self._users):
            if self._user_features is not None:
                encoded_features = [
                    self._user_encodings[_] for _ in self._user_features[i]
                ]
                yield np.asarray([self._user_encodings[user], *encoded_features])
            else:
                yield np.asarray([self._user_encodings[user]])

    @property
    def encoded_items(self):
        for i, item in enumerate(self._items):
            if self._item_features is not None:
                encoded_features = [
                    self._item_encodings[_] for _ in self._item_features[i]
                ]
                yield np.asarray([self._item_encodings[item], *encoded_features])
            else:
                yield np.asarray([self._item_encodings[item]])

    def build(self, negative_samples=0):
        users = np.asarray(list(self.encoded_users))
        items = np.asarray(list(self.encoded_items))

        unique_items = np.unique(items, axis=0).reshape(-1, 1)
        unique_item_ids = unique_items[:, 0]

        for user in users:
            mask = users == user[0]
            user_items = np.unique(items[mask[:, 0]][:, 0], axis=0)
            for (item, rating) in zip(items[mask], self._ratings[mask.flatten()]):
                yield user, np.atleast_1d(item), rating

                if negative_samples > 0:

                    samples = 0

                    while samples < negative_samples:
                        sampled_item = np.random.choice(unique_item_ids)

                        if sampled_item not in user_items:
                            item_mask = unique_items[:, 0] == sampled_item
                            sampled_item_vector = unique_items[item_mask][0]
                            yield user, sampled_item_vector, 0.0
                            samples += 1
