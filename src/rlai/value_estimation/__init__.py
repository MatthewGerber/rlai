import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional, Iterable, Tuple, List, Any, Iterator

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi import PolicyImprovementEvent
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState
from rlai.utils import get_base_argument_parser, log_with_border


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

    @abstractmethod
    def __len__(
            self
    ) -> int:
        """
        Get number of actions defined by the estimator.

        :return: Number of actions.
        """

    @abstractmethod
    def __iter__(
            self
    ) -> Iterator[Action]:
        """
        Get iterator over actions.

        :return: Iterator.
        """

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

        parser = get_base_argument_parser()

        parser.add_argument(
            '--epsilon',
            type=float,
            help='Total probability mass to spread across all actions. Omit or pass 0.0 to produce a purely greedy agent.'
        )

        return parser

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: MdpEnvironment
    ) -> Tuple[Any, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

    @abstractmethod
    def get_initial_policy(
            self
    ) -> Policy:
        """
        Get the initial policy defined by the estimator.

        :return: Policy.
        """

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset the estimator for a new run.

        :param state: Initial state.
        """

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

    @abstractmethod
    def improve_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            event: PolicyImprovementEvent
    ):
        """
        Improve an agent's policy using the current state-action value estimates.

        :param agent: Agent whose policy should be improved.
        :param states: States to improve, or None for all states.
        :param event: Event that triggered the improvement.
        :return: Number of states improved.
        """

        log_with_border(logging.DEBUG, 'Improving policy')

        if event == PolicyImprovementEvent.FINISHED_EVALUATION:
            self.evaluation_policy_improvement_count += 1
        elif event == PolicyImprovementEvent.UPDATED_VALUE_ESTIMATE:
            self.value_estimate_policy_improvement_count += 1
        elif event == PolicyImprovementEvent.MAKING_POLICY_GREEDY:
            pass
        else:  # pragma no cover
            raise ValueError(f'Unknown policy improvement event:  {event}')

    def plot(
            self,
            final: bool,
            pdf: Optional[PdfPages]
    ) -> Optional[plt.Figure]:
        """
        Plot the estimator. If called from the main thread, then the rendering schedule will be checked and a new plot
        will be generated per the schedule. If called from a background thread, then the data used by the plot will be
        updated but a plot will not be generated or updated. This supports a pattern in which a background thread
        generates new plot data, and a UI thread (e.g., in a Jupyter notebook) periodically calls `update_plot` to
        redraw the plot with the latest data.

        :param final: Whether or not this is the final time plot will be called.
        :param pdf: PDF for plots, or None for no PDF.
        :return: Matplotlib figure, if one was generated and not plotting to PDF.
        """

    def update_plot(
            self,
            time_step_detail_iteration: Optional[int]
    ):
        """
        Update the plot of the estimator. Can only be called from the main thread.

        :param time_step_detail_iteration: Iteration for which to plot time-step-level detail, or None for no detail.
        Passing -1 will plot detail for the most recently completed iteration.
        """

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
        elif epsilon < 0.0:
            raise ValueError('epsilon must be >= 0')

        self.epsilon = epsilon

        self.update_count = 0
        self.evaluation_policy_improvement_count: int = 0
        self.value_estimate_policy_improvement_count: int = 0

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
        :return: True if defined and False otherise.
        """

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
