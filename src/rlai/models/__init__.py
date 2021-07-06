from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, List, Any, Optional

import numpy as np
from numpy.random import RandomState

from rlai.meta import rl_text
from rlai.utils import get_base_argument_parser


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

        return get_base_argument_parser()

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
        :return: 2-tuple of a model and a list of unparsed arguments.
        """

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

    @abstractmethod
    def evaluate(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of features.

        :param X: Feature matrix (#obs, #features).
        :return: Vector of outcomes from the evaluation (#obs,).
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
            other
    ) -> bool:
        """
        Check whether the model equals another.

        :param other: Other model.
        :return: True if equal and False otherwise.
        """

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
