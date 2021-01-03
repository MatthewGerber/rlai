from argparse import ArgumentParser
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDRegressor

from rlai.meta import rl_text
from rlai.utils import parse_arguments
from rlai.value_estimation.function_approximation.models import FunctionApproximationModel, FeatureExtractor
from rlai.value_estimation.function_approximation.models.feature_extraction import (
    StateActionInteractionFeatureExtractor
)


@rl_text(chapter=9, page=200)
class SKLearnSGD(FunctionApproximationModel):
    """
    Wrapper for the sklearn.linear_model.SGDRegressor implemented by scikit-learn.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        return parser

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

        parsed_args, unparsed_args = parse_arguments(cls, args)

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

    def get_summary(
            self,
            feature_extractor: FeatureExtractor
    ) -> pd.DataFrame:
        """
        Get a pandas.DataFrame that summarizes the model (e.g., in terms of its coefficients).

        :param feature_extractor: Feature extractor used to build the model.
        :return: DataFrame.
        """

        if isinstance(feature_extractor, StateActionInteractionFeatureExtractor):
            coefficients = pd.DataFrame(
                data=self.model.coef_.reshape((-1, len(feature_extractor.actions)), order='F'),
                columns=[a.name for a in feature_extractor.actions]
            )

            coefficients['feature_name'] = feature_extractor.get_feature_names()
        else:
            raise ValueError(f'Unknown feature extractor type:  {type(feature_extractor)}')

        return coefficients

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
