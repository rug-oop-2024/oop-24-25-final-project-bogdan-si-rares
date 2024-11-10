
from abc import ABC, abstractmethod
from autoop.core.ml.artifact import Artifact
import numpy as np
from copy import deepcopy


class Model(Artifact, ABC):
    def __init__(
        self,
        name: str,
        data: bytes = b"",
        type: str = "model",
        parameters=None
    ):
        """
        Initialize the Model class, inheriting from Artifact.

        Args:
            name (str): Name of the model.
            data (bytes): Serialized model data (default: empty bytes).
            type (str): Type of the model (default: 'model').
            parameters (dict): Optional model parameters.
        """
        super().__init__(name=name, data=data, type=type)
        self._parameters = parameters or {}  # Initialize _parameters here

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def save(self) -> None:
        self.data = deepcopy(self._parameters)
        self.data["type"] = self.type
        super().save(self.data)

    @classmethod
    def load(cls, artifact: Artifact) -> "Model":
        model_data = deepcopy(artifact.data)
        model_type = model_data.pop("type")
        name = artifact.name
        return cls(
            model_type=model_type,
            parameters=model_data,
            name=name
        )

    @property
    def parameters(self):
        """Get model parameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, new_params):
        """Set or update model parameters."""
        if not hasattr(self, '_parameters'):
            self._parameters = {}  # Ensure _parameters is initialized
        self._parameters.update(new_params)
