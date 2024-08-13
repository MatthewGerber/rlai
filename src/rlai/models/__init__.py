from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, List, Any, Optional

import numpy as np
from numpy.random import RandomState

from rlai.docs import rl_text
from rlai.utils import get_base_argument_parser


@rl_text(chapter=9, page=197)
class FunctionApproximationModel(ABC):
    """
    Base class for models that approximate functions.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        return get_base_argument_parser()

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            fit_intercept: bool
    ) -> Tuple['FunctionApproximationModel', List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param fit_intercept: Whether to fit an intercept term.
        :return: 2-tuple of a model and a list of unparsed arguments.
        """

    @abstractmethod
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

    @abstractmethod
    def evaluate(
            self,
            feature_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of features.

        :param feature_matrix: Feature matrix (#obs, #features).
        :return: Vector of outcomes from the evaluation (#obs,).
        """

    def reset(
            self
    ):
        """
        Reset the model.
        """

    def __init__(
            self
    ):
        """
        Initialize the model.
        """

    @abstractmethod
    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the model equals another.

        :param other: Other model.
        :return: True if equal and False otherwise.
        """

    @abstractmethod
    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the model does not equal another.

        :param other: Other model.
        :return: True if not equal and False otherwise.
        """
