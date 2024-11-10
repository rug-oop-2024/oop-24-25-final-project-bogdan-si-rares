from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Dict
from autoop.core.ml.model import Model


class LinearRegressionModel(Model):
    def __init__(self, parameters: Dict[str, any] = None):
        super().__init__(model_type="regression", parameters=parameters)
        self._model = LinearRegression(**self.parameters)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)
