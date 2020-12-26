from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor

from rlai.meta import rl_text
from rlai.value_estimation.function_approximation.statistical_learning import FunctionApproximationModel


@rl_text(chapter=9, page=200)
class SKLearnSGD(FunctionApproximationModel):
    """
    Wrapper for the sklearn.linear_model.SGDRegressor implemented by scikit-learn.
    """

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parsed_args, unparsed_args = super().parse_arguments(args)

        parser = ArgumentParser(allow_abbrev=False)

        # future arguments to be added here...

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str]
    ) -> Tuple[FunctionApproximationModel, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        model = SKLearnSGD(
            **vars(parsed_args)
        )

        return model, unparsed_args

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            weight: Optional[float]
    ):
        """
        Fit the model to a matrix of features (one row per observations) and a vector of returns.

        :param X: Feature matrix (#obs x #features)
        :param y: Outcome vector (#obs).
        :param weight: Weight.
        """

        self.model.partial_fit(X=X, y=y, sample_weight=weight)

    def evaluate(
            self,
            X: np.ndarray
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of features (one row per observation).

        :param X: Feature matrix (#obs x #features).
        :return: Vector of outcomes from the evaluation (#obs).
        """

        try:

            # predict values using the currently fitted model
            values = self.model.predict(X)

        # the following exception will be thrown if the model has not yet been fitted. catch and return uniformly valued
        # outcomes.
        except NotFittedError:
            values = np.repeat(0.0, X.shape[0])

        return values

    def __init__(
            self,
            **kwargs
    ):
        """
        Initialize the model.

        :param kwargs: Keyword arguments to pass to SGDRegressor.
        """

        self.model = SGDRegressor(**kwargs)

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the model equals another.

        :param other: Other model.
        :return: True if equal and False otherwise.
        """

        return np.array_equal(self.model.coef_, other.model.coef_)

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the model does not equal another.

        :param other: Other model.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
