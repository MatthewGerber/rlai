from typing import Optional, List, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.models import FunctionApproximationModel
from rlai.models.sklearn import SKLearnSGD as SKLearnSGDRegressor
from rlai.state_value.function_approximation.models import StateFunctionApproximationModel


class SKLearnSGD(StateFunctionApproximationModel):
    """
    State-action value modeler based on the SKLearnSGD algorithm.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            fit_intercept: bool
    ) -> Tuple[FunctionApproximationModel, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param fit_intercept: Whether to fit an intercept term.
        :return: 2-tuple of a model and a list of unparsed arguments.
        """

        sklearn_sgd, unparsed_args = SKLearnSGDRegressor.init_from_arguments(
            args=args,
            random_state=random_state,
            fit_intercept=fit_intercept
        )

        assert isinstance(sklearn_sgd, SKLearnSGDRegressor)

        state_action_sklearn_sgd = cls(
            sklearn_sgd=sklearn_sgd
        )

        return state_action_sklearn_sgd, unparsed_args

    def fit(
            self,
            feature_matrix: Any,
            outcomes: Any,
            weights: Optional[np.ndarray]
    ):
        """
        Fit the model to a matrix of feature vectors and a vector of outcomes.

        :param feature_matrix: Feature matrix (#obs, #features).
        :param outcomes: Outcome vector (#obs,).
        :param weights: Weights (#obs,).
        """

        self.sklearn_sgd.fit(feature_matrix, outcomes, weights)

    def evaluate(
            self,
            feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of features.

        :param feature_matrix: Feature matrix (#obs, #features).
        :return: Vector of outcomes from the evaluation (#obs,).
        """

        return self.sklearn_sgd.evaluate(feature_matrix)

    def plot(
            self,
            render: bool,
            pdf: Optional[PdfPages]
    ) -> Optional[plt.Figure]:
        """
        Plot the model. If called from the main thread and render is True, then a new plot will be generated. If called
        from a background thread, then the data used by the plot will be updated but a plot will not be generated or
        updated. This supports a pattern in which a background thread generates new plot data, and a UI thread (e.g., in
        a Jupyter notebook) periodically calls `update_plot` to redraw the plot with the latest data.

        :param render: Whether to render the plot data. If False, then plot data will be updated but nothing will
        be shown.
        :param pdf: PDF for plots.
        :return: Matplotlib figure, if one was generated and not plotting to PDF.
        """

        return self.sklearn_sgd.plot(
            render=render,
            pdf=pdf
        )

    def __init__(
            self,
            sklearn_sgd: SKLearnSGDRegressor
    ):
        """
        Initialize the model.

        :param sklearn_sgd: SKLearnSGD instance.
        """

        super().__init__()

        self.sklearn_sgd = sklearn_sgd

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check equality.

        :param other: Other object.
        :return: True if equal.
        """

        if not isinstance(other, SKLearnSGD):
            raise ValueError(f'Expected a {SKLearnSGD}')

        return self.sklearn_sgd.__eq__(other.sklearn_sgd)

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check inequality.

        :param other: Other object.
        :return: True if not equal.
        """

        if not isinstance(other, SKLearnSGD):
            raise ValueError(f'Expected a {SKLearnSGD}')

        return self.sklearn_sgd.__ne__(other.sklearn_sgd)
