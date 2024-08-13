from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional, Tuple, List

from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.core import MdpState, Environment
from rlai.docs import rl_text
from rlai.utils import get_base_argument_parser


@rl_text(chapter='Value Estimation', page=23)
class ValueEstimator(ABC):
    """
    Value estimator.
    """

    @abstractmethod
    def update(
            self,
            value: float,
            weight: Optional[float] = None
    ):
        """
        Update the value estimate.

        :param value: New value.
        :param weight: Weight.
        """

    @abstractmethod
    def get_value(
            self
    ) -> float:
        """
        Get current estimated value.

        :return: Value.
        """

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return str(self.get_value())


@rl_text(chapter='Value Estimation', page=23)
class StateValueEstimator(ABC):
    """
    State-value estimator.
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
            environment: Environment
    ) -> Tuple['StateValueEstimator', List[str]]:
        """
        Initialize a state-value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a state-value estimator and a list of unparsed arguments.
        """

    @abstractmethod
    def improve(
            self
    ):
        """
        Improve the estimator.
        """

    def plot(
            self,
            pdf: Optional[PdfPages]
    ):
        """
        Plot the current estimator.

        :param pdf: PDF to plot to, or None to show directly.
        """

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset for new run.
        """

    def __init__(
            self
    ):
        """
        Initialize the estimator.
        """

        self.update_count = 0

    @abstractmethod
    def __getitem__(
            self,
            state: MdpState
    ) -> ValueEstimator:
        """
        Get the value estimator for a state.

        :param state: State.
        :return: Value estimator.
        """

    @abstractmethod
    def __len__(
            self
    ) -> int:
        """
        Get number of states defined by the estimator.

        :return: Number of states.
        """

    @abstractmethod
    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether a state is defined by the estimator.

        :param state: State.
        :return: True if defined and False otherwise.
        """

    @abstractmethod
    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """

    @abstractmethod
    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """
