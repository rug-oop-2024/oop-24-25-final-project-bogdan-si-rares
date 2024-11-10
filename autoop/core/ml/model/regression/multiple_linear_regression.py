from sklearn.linear_model import LinearRegression
import numpy as np
from autoop.core.ml.model import Model


class MultipleLinearRegression(Model):
    def __init__(self, parameters=None):
        # Initialize required fields for Artifact
        super().__init__(
            name="MultipleLinearRegression",
            data=b"",  # Provide valid bytes
            type="regression",  # Specify type
            parameters=parameters,
        )
        self._internal_model = LinearRegression(**(parameters or {}))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """Fit the linear regression model."""
        self._internal_model.fit(x, y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using the linear regression model."""
        return self._internal_model.predict(x)
