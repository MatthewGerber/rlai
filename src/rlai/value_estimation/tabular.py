from argparse import ArgumentParser
from typing import Dict, Optional, Iterable, Iterator, List, Tuple

from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.meta import rl_text
from rlai.policies.tabular import TabularPolicy
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager, parse_arguments
from rlai.value_estimation import StateActionValueEstimator, ActionValueEstimator, ValueEstimator


@rl_text(chapter='Value Estimation', page=23)
class TabularValueEstimator(ValueEstimator):
    """
    Tabular value estimator.
    """

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

        self.averager.update(
            value=value,
            weight=weight
        )

        self.estimator.update_count += 1

    def get_value(
            self
    ) -> float:
        """
        Get current estimated value.

        :return: Value.
        """

        return self.averager.get_value()

    def __init__(
            self,
            estimator,
            alpha: float,
            weighted: bool
    ):
        """
        Initialize the estimator.

        :param estimator: Estimator
        :param alpha: Step size.
        :param weighted: Whether estimator should be weighted.
        """

        self.estimator: TabularStateActionValueEstimator = estimator

        self.averager = IncrementalSampleAverager(
            alpha=alpha,
            weighted=weighted
        )

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if estimates are equal and False otherwise.
        """

        other: TabularValueEstimator

        return self.averager == other.averager

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if estimates are not equal and False otherwise.
        """

        return not (self == other)


@rl_text(chapter='Value Estimation', page=23)
class TabularActionValueEstimator(ActionValueEstimator):
    """
    Tabular action-value estimator.
    """

    def __init__(
            self,
            estimator
    ):
        """
        Initialize the estimator.

        :param estimator: Estimator.
        """

        self.estimator: TabularStateActionValueEstimator = estimator

        self.q_A: Dict[Action, TabularValueEstimator] = {}

    def __contains__(
            self,
            action: Action
    ) -> bool:
        """
        Check whether action is defined.

        :param action: Action.
        :return: True if defined and False otherwise.
        """

        return action in self.q_A

    def __getitem__(
            self,
            action: Action
    ) -> TabularValueEstimator:
        """
        Get value estimator for an action.

        :param action: Action.
        :return: Value estimator.
        """

        return self.q_A[action]

    def __setitem__(
            self,
            action: Action,
            value_estimator: TabularValueEstimator
    ):
        """
        Set the estimator for an action.

        :param action: Action.
        :param value_estimator: Estimator.
        """

        self.q_A[action] = value_estimator

    def __len__(
            self
    ) -> int:
        """
        Get number of actions defined by the estimator.

        :return: Number of actions.
        """

        return len(self.q_A)

    def __iter__(
            self
    ) -> Iterator[Action]:
        """
        Get iterator over actions.

        :return: Iterator.
        """

        return iter(self.q_A)

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if estimates are equal and False otherwise.
        """

        other: TabularActionValueEstimator

        return self.q_A == other.q_A

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if estimates are not equal and False otherwise.
        """

        return not (self == other)


@rl_text(chapter='Value Estimation', page=23)
class TabularStateActionValueEstimator(StateActionValueEstimator):
    """
    Tabular state-action value estimator.
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
            '--continuous-state-discretization-resolution',
            type=float,
            help='Continuous-state discretization resolution.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: MdpEnvironment,
            epsilon: float
    ) -> Tuple[StateActionValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :param epsilon: Epsilon.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        estimator = TabularStateActionValueEstimator(
            environment=environment,
            epsilon=epsilon,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

    def get_initial_policy(
            self
    ) -> TabularPolicy:
        """
        Get the initial policy defined by the estimator.

        :return: Policy.
        """

        return TabularPolicy(
            continuous_state_discretization_resolution=self.continuous_state_discretization_resolution,
            SS=self.SS
        )

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
        """

        if state not in self:
            self[state] = TabularActionValueEstimator(estimator=self)

        if a not in self[state]:
            self[state][a] = TabularValueEstimator(estimator=self, alpha=alpha, weighted=weighted)

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

        if epsilon is None:
            epsilon = 0.0

        self.epsilon = epsilon

        q_pi = {
            s: {
                a: self[s][a].get_value()
                for a in self[s]
            }
            for s in self
            if states is None or s in states
        }

        num_states_improved = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi,
            epsilon=self.epsilon
        )

        return num_states_improved

    def __init__(
            self,
            environment: MdpEnvironment,
            epsilon: Optional[float],
            continuous_state_discretization_resolution: Optional[float]
    ):
        """
        Initialize the estimator.

        :param environment: Environment.
        :param epsilon: Epsilon, or None for a purely greedy policy.
        :param continuous_state_discretization_resolution: A discretization resolution for continuous-state
        environments. Providing this value allows the agent to be used with discrete-state methods via
        discretization of the continuous-state dimensions.
        """

        super().__init__(
            environment=environment,
            epsilon=epsilon
        )

        self.SS = environment.SS
        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution

        self.q_S_A: Dict[MdpState, TabularActionValueEstimator] = {}

        # for completeness, initialize the estimator for all terminal states. these will not be updated during execution
        # since no action ever takes an agent out of them; however, terminal states should have a value represented, if
        # only ever it is zero.
        for terminal_state in environment.terminal_states:
            for a in terminal_state.AA:
                self.initialize(
                    state=terminal_state,
                    a=a,
                    alpha=None,
                    weighted=False
                )

    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether a state is defined by the estimator.

        :param state: State.
        :return: True if defined and False otherise.
        """

        return state in self.q_S_A

    def __getitem__(
            self,
            state: MdpState
    ) -> TabularActionValueEstimator:
        """
        Get the action-value estimator for a state.

        :param state: State.
        :return: Action-value estimator.
        """

        return self.q_S_A[state]

    def __setitem__(
            self,
            state: MdpState,
            action_value_estimator: TabularActionValueEstimator
    ):
        """
        Set the action-value estimator for a state.

        :param state: State.
        :param action_value_estimator: Estimator.
        """

        self.q_S_A[state] = action_value_estimator

    def __len__(
            self
    ) -> int:
        """
        Get number of states defined by the estimator.

        :return: Number of states.
        """

        return len(self.q_S_A)

    def __iter__(
            self
    ) -> Iterator:
        """
        Get iterator over state-action items.

        :return: State-action items.
        """

        return iter(self.q_S_A)

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """

        other: TabularStateActionValueEstimator

        return self.q_S_A == other.q_S_A

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
