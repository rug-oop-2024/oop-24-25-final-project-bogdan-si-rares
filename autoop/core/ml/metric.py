from abc import ABC, abstractmethod
import numpy as np


METRICS = [
    "mean_squared_error",
    "accuracy",
    "mean_absolute_error",
    "r_squared",
    "precision",
    "recall",
]  # add the names (in strings) of the metrics you implement


def get_metric(name: str):
    # Factory function to get a metric by name.
    # Return a metric instance given its str name.
    if name == "mean_squared_error":
        return MeanSquaredError()
    elif name == "accuracy":
        return Accuracy()
    elif name == "mean_absolute_error":
        return MeanAbsoluteError()
    elif name == "r_squared":
        return RSquared()
    elif name == "precision":
        return Precision()
    elif name == "recall":
        return Recall()
    else:
        raise ValueError(f"Unknown metric: {name}")


class Metric(ABC):
    """Base class for all metrics.
    """

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate the metric using the provided ground truth and predictions.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The computed metric value.
        """
        pass

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        A standardized method for evaluation, calling the __call__ method.

        Args:
            y_true (np.ndarray): Ground truth values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The computed metric value.
        """
        return self(y_true, y_pred)


class MeanSquaredError(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)


class Accuracy(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(y_true == y_pred)


class MeanAbsoluteError(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))


class RSquared(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total)


class Precision(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        predicted_positives = np.sum(y_pred == 1)
        return (
            true_positives / predicted_positives if predicted_positives > 0
            else 0.0
        )


class Recall(Metric):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        actual_positives = np.sum(y_true == 1)
        return (
            true_positives / actual_positives if actual_positives > 0 else 0.0
        )
