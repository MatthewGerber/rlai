import numpy as np
from sklearn.linear_model import SGDRegressor

from rlai.value_estimation.function_approximation.models import FunctionApproximationModel


class SKLearnSGD(FunctionApproximationModel):

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            weight: float
    ):
        self.model.partial_fit(X=X, y=y, sample_weight=weight)

    def evaluate(
            self,
            X: np.ndarray
    ) -> np.ndarray:

        return self.model.predict(X)

    def __init__(
            self,
            **kwargs
    ):
        self.model = SGDRegressor(**kwargs)
