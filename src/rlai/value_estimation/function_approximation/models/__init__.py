from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, List, Any, Optional

import numpy as np
import pandas as pd
from numpy.random import RandomState

from rlai.meta import rl_text
from rlai.value_estimation.function_approximation.models.feature_extraction import FeatureExtractor


@rl_text(chapter=9, page=197)
class FunctionApproximationModel(ABC):
    """
    Function approximation model.
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
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--help',
            action='store_true',
            help='Print usage and argument descriptions.'
        )

        return parser

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Any, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """
        pass

    @abstractmethod
    def fit(
            self,
            X: Any,
            y: Any,
            weight: Optional[float]
    ):
        """
        Fit the model to a matrix of features and a vector of returns.

        :param X: Feature matrix.
        :param y: Returns.
        :param weight: Weight.
        """
        pass

    @abstractmethod
    def evaluate(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of observations.

        :param X: Observations.
        :return: Vector of evaluations.
        """
        pass

    @abstractmethod
    def get_summary(
            self,
            feature_extractor: FeatureExtractor
    ) -> pd.DataFrame:
        """
        Get a pandas.DataFrame that summarizes the model (e.g., in terms of its coefficients).

        :param feature_extractor: Feature extractor used to build the model.
        :return: DataFrame.
        """
        pass

    @abstractmethod
    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the model equals another.

        :param other: Other model.
        :return: True if equal and False otherwise.
        """
        pass

    @abstractmethod
    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the model does not equal another.

        :param other: Other model.
        :return: True if not equal and False otherwise.
        """
        pass
