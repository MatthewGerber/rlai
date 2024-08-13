import logging
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Optional, Iterator, List, Tuple, Any, Iterable

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.core import Policy, Action, MdpState, Agent, StochasticMdpAgent, Environment
from rlai.core.environments.mdp import MdpEnvironment
from rlai.docs import rl_text
from rlai.gpi import PolicyImprovementEvent
from rlai.utils import get_base_argument_parser, log_with_border, parse_arguments, load_class


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
    ) -> Tuple['StateActionValueEstimator', List[str]]:
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
            agent: Any,
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

        :param final: Whether this is the final time plot will be called.
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
            epsilon: Optional[float],
            **_
    ):
        """
        Initialize the estimator.

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


@rl_text(chapter='Agents', page=1)
class ActionValueMdpAgent(StochasticMdpAgent):
    """
    A stochastic MDP agent whose policy is based on action-value estimation. This agent is generally appropriate for
    discrete and continuous state spaces in which we estimate the value of actions using tabular and
    function-approximation methods, respectively. The action space need to be discrete in all of these cases. If the
    action space is continuous, then consider the `ParameterizedMdpAgent`.
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

        parser.add_argument(
            '--q-S-A',
            type=str,
            help='Fully-qualified type name of state-action value estimator to use.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: Environment
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize an MDP agent from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # load state-action value estimator
        estimator_class = load_class(parsed_args.q_S_A)
        q_S_A, unparsed_args = estimator_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            environment=environment
        )
        del parsed_args.q_S_A

        # noinspection PyUnboundLocalVariable
        agent = cls(
            name=f'action-value (gamma={parsed_args.gamma})',
            random_state=random_state,
            q_S_A=q_S_A,
            **vars(parsed_args)
        )

        return [agent], unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            gamma: float,
            q_S_A: StateActionValueEstimator
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param gamma: Discount.
        :param q_S_A: State-action value estimator.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            pi=q_S_A.get_initial_policy(),
            gamma=gamma
        )

        self.q_S_A = q_S_A
