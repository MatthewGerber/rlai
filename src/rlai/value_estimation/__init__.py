from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional, Iterable, Tuple, List, Any, Iterator

from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState
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
        pass

    @abstractmethod
    def get_value(
            self
    ) -> float:
        """
        Get current estimated value.

        :return: Value.
        """
        pass

    def __str__(
            self
    ) -> str:
        """
        String override.

        :return: String.
        """

        return str(self.get_value())


@rl_text(chapter='Value Estimation', page=23)
class ActionValueEstimator(ABC):
    """
    Action value estimator.
    """

    @abstractmethod
    def __getitem__(
            self,
            action: Action
    ) -> ValueEstimator:
        """
        Get value estimator for an action.

        :param action: Action.
        :return: Value estimator.
        """
        pass

    @abstractmethod
    def __len__(
            self
    ) -> int:
        """
        Get number of actions defined by the estimator.

        :return: Number of actions.
        """
        pass

    @abstractmethod
    def __iter__(
            self
    ) -> Iterator[Action]:
        """
        Get iterator over actions.

        :return: Iterator.
        """
        pass

    @abstractmethod
    def __contains__(
            self,
            action: Action
    ) -> bool:
        """
        Check whether action is defined.

        :param action: Action.
        :return: True if defined and False otherwise.
        """
        pass


@rl_text(chapter='Value Estimation', page=23)
class StateActionValueEstimator(ABC):
    """
    State-action value estimator.
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
            environment: MdpEnvironment,
            epsilon: float
    ) -> Tuple[Any, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :param epsilon: Epsilon.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """
        pass

    @abstractmethod
    def get_initial_policy(
            self
    ) -> Policy:
        """
        Get the initial policy defined by the estimator.

        :return: Policy.
        """
        pass

    def initialize(
            self,
            state: MdpState,
            a: Action,
            alpha: Optional[float],
            weighted: bool
    ):
        """
        Initialize the estimator for a state-action pair.

        :param state: State.
        :param a: Action.
        :param alpha: Step size.
        :param weighted: Whether the estimator should be weighted.
        :return:
        """
        pass

    @abstractmethod
    def improve_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            epsilon: float
    ) -> int:
        """
        Improve an agent's policy using the current state-action value estimates.

        :param agent: Agent whose policy should be improved.
        :param states: States to improve, or None for all states.
        :param epsilon: Epsilon.
        :return: Number of states improved.
        """
        pass

    def plot(
            self,
            final: bool
    ):
        """
        Plot the estimator.

        :param final: Whether or not this is the final time plot will be called.
        """
        pass

    def __init__(
            self,
            environment: MdpEnvironment,
            epsilon: Optional[float]
    ):
        """
        Initialize the estimator.

        :param environment: Environment.
        :param epsilon: Epsilon, or None for a purely greedy policy.
        """

        if epsilon is None:
            epsilon = 0.0

        self.epsilon = epsilon

        self.update_count = 0

    @abstractmethod
    def __getitem__(
            self,
            state: MdpState
    ) -> ActionValueEstimator:
        """
        Get the action-value estimator for a state.

        :param state: State.
        :return: Action-value estimator.
        """
        pass

    @abstractmethod
    def __len__(
            self
    ) -> int:
        """
        Get number of states defined by the estimator.

        :return: Number of states.
        """
        pass

    @abstractmethod
    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether a state is defined by the estimator.

        :param state: State.
        :return: True if defined and False otherise.
        """
        pass

    @abstractmethod
    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """
        pass

    @abstractmethod
    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """
        pass
