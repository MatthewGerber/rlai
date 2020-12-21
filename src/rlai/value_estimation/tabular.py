from argparse import Namespace, ArgumentParser
from typing import Dict, Optional, Iterable, Iterator, List, Tuple

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.meta import rl_text
from rlai.policies.tabular import TabularPolicy
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager
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
        self.averager.update(
            value=value,
            weight=weight
        )

    def get_value(
            self
    ) -> float:

        return self.averager.get_value()

    def __init__(
            self,
            alpha: float,
            weighted: bool
    ):
        self.averager = IncrementalSampleAverager(
            alpha=alpha,
            weighted=weighted
        )

    def __eq__(
            self,
            other
    ) -> bool:

        other: TabularValueEstimator

        return self.averager == other.averager

    def __ne__(
            self,
            other
    ) -> bool:
        return not (self == other)


@rl_text(chapter='Value Estimation', page=23)
class TabularActionValueEstimator(ActionValueEstimator):
    """
    Tabular action-value estimator.
    """

    def __init__(
            self
    ):
        self.q_A: Dict[Action, TabularValueEstimator] = {}

    def __contains__(
            self,
            action: Action
    ) -> bool:

        return action in self.q_A

    def __getitem__(
            self,
            action: Action
    ) -> TabularValueEstimator:

        return self.q_A[action]

    def __setitem__(
            self,
            action: Action,
            value_estimator: TabularValueEstimator
    ):
        self.q_A[action] = value_estimator

    def __len__(
            self
    ) -> int:

        return len(self.q_A)

    def __iter__(
            self
    ) -> Iterator[Action]:

        return self.q_A.__iter__()

    def __eq__(
            self,
            other
    ) -> bool:

        other: TabularActionValueEstimator

        return self.q_A == other.q_A

    def __ne__(
            self,
            other
    ) -> bool:
        return not (self == other)


@rl_text(chapter='Value Estimation', page=23)
class TabularStateActionValueEstimator(StateActionValueEstimator):
    """
    Tabular state-action value estimator.
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

        parser.add_argument(
            '--continuous-state-discretization-resolution',
            type=float,
            help='Continuous-state discretization resolution.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[StateActionValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        estimator = TabularStateActionValueEstimator(
            environment=environment,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

    def get_initial_policy(
            self
    ) -> TabularPolicy:

        return TabularPolicy(
            continuous_state_discretization_resolution=self.continuous_state_discretization_resolution,
            SS=self.environment.SS
        )

    def initialize(
            self,
            state: MdpState,
            a: Action,
            alpha: Optional[float],
            weighted: bool
    ):
        if state not in self:
            self[state] = TabularActionValueEstimator()

        if a not in self[state]:
            self[state][a] = TabularValueEstimator(alpha=alpha, weighted=weighted)

    def update_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            epsilon: float
    ) -> int:

        q_pi = {
            s: {
                a: self[s][a].get_value()
                for a in self[s]
            }
            for s in self
            if states is None or s in states
        }

        num_states_updated = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi,
            epsilon=epsilon
        )

        return num_states_updated

    def __init__(
            self,
            environment: MdpEnvironment,
            continuous_state_discretization_resolution: Optional[float]
    ):
        """
        Initialize the estimator.

        :param environment: Environment.
        :param continuous_state_discretization_resolution: A discretization resolution for continuous-state
        environments. Providing this value allows the agent to be used with discrete-state methods via
        discretization of the continuous-state dimensions.
        """

        self.environment = environment
        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution

        self.q_S_A: Dict[MdpState, TabularActionValueEstimator] = {}

        # for completeness, initialize the estimator for all terminal states. these will not be updated during execution
        # since no action ever takes an agent out of them; however, terminal states should have a value represented, if
        # only ever it is zero.
        for terminal_state in self.environment.terminal_states:
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

        return state in self.q_S_A

    def __getitem__(
            self,
            state: MdpState
    ) -> TabularActionValueEstimator:

        return self.q_S_A[state]

    def __setitem__(
            self,
            state: MdpState,
            action_value_estimator: TabularActionValueEstimator
    ):
        self.q_S_A[state] = action_value_estimator

    def __len__(
            self
    ) -> int:

        return len(self.q_S_A)

    def __iter__(
            self
    ) -> Iterator[MdpState]:

        return self.q_S_A.__iter__()

    def __eq__(
            self,
            other
    ) -> bool:

        other: TabularStateActionValueEstimator

        return self.q_S_A == other.q_S_A

    def __ne__(
            self,
            other
    ) -> bool:

        return not (self == other)
