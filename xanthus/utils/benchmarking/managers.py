from typing import Any, Dict, Tuple
from abc import ABC, abstractmethod


class ModelManager(ABC):
    def __init__(self, name: str, datasets: Tuple, **params: Any) -> None:
        self.name = name
        self._params = params
        self.train = datasets[0]
        self.val = datasets[1]

    @abstractmethod
    def update(self, epochs: int) -> None:
        pass

    @abstractmethod
    def metrics(self, **kwargs) -> Dict[str, float]:
        pass

    def params(self):
        return self._params
