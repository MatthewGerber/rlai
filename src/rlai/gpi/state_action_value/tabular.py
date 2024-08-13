import logging
from argparse import ArgumentParser
from typing import Optional, Dict, Iterator, List, Tuple, Iterable, Union

import numpy as np
from numpy.random import RandomState

from rlai.core import Policy, Action, MdpState, MdpAgent
from rlai.core.environments.mdp import MdpEnvironment, ModelBasedMdpEnvironment
from rlai.docs import rl_text
from rlai.gpi import PolicyImprovementEvent
from rlai.gpi.state_action_value import ValueEstimator, ActionValueEstimator, StateActionValueEstimator
from rlai.utils import IncrementalSampleAverager, parse_arguments, log_with_border


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
            estimator: 'TabularStateActionValueEstimator',
            alpha: Optional[float],
            weighted: bool
    ):
        """
        Initialize the estimator.

        :param estimator: Estimator
        :param alpha: Step size.
        :param weighted: Whether estimator should be weighted.
        """

        self.estimator = estimator

        self.averager = IncrementalSampleAverager(
            alpha=alpha,
            weighted=weighted
        )

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if estimates are equal and False otherwise.
        """

        if not isinstance(other, TabularValueEstimator):
            raise ValueError(f'Expected {TabularValueEstimator}')

        return self.averager == other.averager

    def __ne__(
            self,
            other: object
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
            estimator: 'TabularStateActionValueEstimator'
    ):
        """
        Initialize the estimator.

        :param estimator: Estimator.
        """

        self.estimator = estimator

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
            other: object
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if estimates are equal and False otherwise.
        """

        if not isinstance(other, TabularActionValueEstimator):
            raise ValueError(f'Expected {TabularActionValueEstimator}')

        return self.q_A == other.q_A

    def __ne__(
            self,
            other: object
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
            environment: MdpEnvironment
    ) -> Tuple[StateActionValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        estimator = cls(
            environment=environment,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

    def get_initial_policy(
            self
    ) -> 'TabularPolicy':
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
            action_value_estimator = self[state]
            assert isinstance(action_value_estimator, TabularActionValueEstimator)
            action_value_estimator[a] = TabularValueEstimator(estimator=self, alpha=alpha, weighted=weighted)

    def improve_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            event: PolicyImprovementEvent
    ) -> int:
        """
        Improve an agent's policy using the current state-action value estimates.

        :param agent: Agent whose policy should be improved.
        :param states: States to improve, or None for all states.
        :param event: Event that triggered the improvement.
        :return: Number of states improved.
        """

        super().improve_policy(
            agent=agent,
            states=states,
            event=event
        )

        q_pi = {
            s: {
                a: self[s][a].get_value()
                for a in self[s]
            }
            for s in self
            if states is None or s in states
        }

        assert isinstance(agent.pi, TabularPolicy)

        num_states_improved = agent.pi.improve_with_q_pi(
            q_pi=q_pi,
            epsilon=self.epsilon
        )

        log_with_border(logging.DEBUG, 'Policy improved')

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

        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution
        self.SS = environment.SS
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
        :return: True if defined and False otherwise.
        """

        return state in self.q_S_A

    def __getitem__(
            self,
            state: MdpState
    ) -> ActionValueEstimator:
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
            other: object
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, TabularStateActionValueEstimator):
            raise ValueError(f'Expected {TabularStateActionValueEstimator}')

        return self.q_S_A == other.q_S_A

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)


@rl_text(chapter=3, page=58)
class TabularPolicy(Policy):
    """
    Policy for use with tabular methods.
    """

    def get_state_i(
            self,
            state_descriptor: Union[str, np.ndarray]
    ) -> Optional[int]:
        """
        Get the integer identifier for a state. The returned value is guaranteed to be the same for the same state,
        both throughout the life of the current agent and after the current agent has been pickled for later use (e.g.,
        in checkpoint-based resumption).

        :param state_descriptor: State descriptor, either a string (for discrete states) or an array representing a
        position within an n-dimensional continuous state space, which will be discretized.
        :return: Integer identifier.
        """

        # build a unique state string from a multidimensional state array in the case of discretized continuous states
        if isinstance(state_descriptor, np.ndarray):

            if self.continuous_state_discretization_resolution is None:
                raise ValueError('Attempted to discretize a continuous state without a resolution.')

            state_descriptor = '|'.join(
                str(int(state_dim_value / self.continuous_state_discretization_resolution))
                for state_dim_value in state_descriptor
            )

        elif not isinstance(state_descriptor, str):
            raise ValueError(f'Unknown state space type:  {type(state_descriptor)}')

        # lazy-initialize the state identifier. this will grow unbounded with increasing resolution of a discretized
        # continuous state. tabular policies aren't well suited to such environments. function approximation is better.
        # but we can still try, and if memory requirements aren't too great this will work okay.
        if state_descriptor not in self.state_id_str_int:
            self.state_id_str_int[state_descriptor] = len(self.state_id_str_int)

        return self.state_id_str_int[state_descriptor]

    def improve_with_v_pi(
            self,
            gamma: float,
            environment: ModelBasedMdpEnvironment,
            v_pi: Dict[MdpState, float]
    ) -> int:
        """
        Improve the current policy according to state-value estimates. This makes the policy greedy with respect to the
        state-value estimates. In cases where multiple such greedy actions exist for a state, each of the greedy actions
        will be assigned equal probability.

        Note that the present function requires state-value estimates of states that are model-based. This is the case
        because policy improvement from state values is only possible if we have a model of the environment. Compare
        with `improve_with_q_pi`, which accepts model-free states since state-action values are estimated directly.

        :param gamma: Discount.
        :param environment: Model-based environment.
        :param v_pi: State-value estimates for the policy.
        :return: Number of states in which the policy was improved.
        """

        # calculate state-action values (q) for the agent's policy
        q_S_A = {
            s: {
                a: sum([
                    environment.p_S_prime_R_given_S_A[s][a][s_prime][r] * (r.r + gamma * v_pi[s_prime])
                    for s_prime in environment.p_S_prime_R_given_S_A[s][a]
                    for r in environment.p_S_prime_R_given_S_A[s][a][s_prime]
                ])
                for a in environment.p_S_prime_R_given_S_A[s]
            }
            for s in self
        }

        return self.improve_with_q_pi(
            q_pi=q_S_A
        )

    def improve_with_q_pi(
            self,
            q_pi: Dict[MdpState, Dict[Action, float]],
            epsilon: Optional[float] = None
    ) -> int:
        """
        Improve the current policy according to state-action value estimates. This makes the policy greedy with respect
        to the state-action value estimates. In cases where multiple such greedy actions exist for a state, each of the
        greedy actions will be assigned equal probability.

        :param q_pi: State-action value estimates for the policy.
        :param epsilon: Total probability mass to divide across all actions for a state, resulting in an epsilon-greedy
        policy. Must be >= 0.0 if given. Pass None to generate a purely greedy policy.
        :return: Number of states in which the policy was improved.
        """

        if epsilon is None:
            epsilon = 0.0
        elif epsilon < 0.0:
            raise ValueError('Epsilon must be >= 0')

        # get the maximal action value for each state
        S_max_q = {
            s: max(q_pi[s].values())
            for s in q_pi
        }

        # count up how many actions in each state are maximizers (i.e., tied in action value)
        S_num_maximizers = {
            s: sum(q_pi[s][a] == S_max_q[s] for a in q_pi[s])
            for s in q_pi
        }

        # generate policy improvement, assigning uniform probability across all maximizing actions in addition to a
        # uniform fraction of epsilon spread across all actions in the state.
        policy_improvement = {
            s: {
                a: (

                    # actions that tie in maximizing value share the non-epsilon probability
                    ((1.0 - epsilon) / S_num_maximizers[s]) +

                    # all actions get an equal share of the epsilon probability
                    (epsilon / len(s.AA))

                    # the above only apply to value maximizers
                    if a in q_pi[s] and q_pi[s][a] == S_max_q[s]

                    # all other actions get a share of the epsilon probability
                    else epsilon / len(s.AA)
                )

                # improve policy for all feasible actions in the state
                for a in s.AA
            }
            for s in self

            # we can only improve the policy for states that we have q-value estimates for
            if s in q_pi
        }

        # count up how many states got a new action distribution
        num_states_improved = sum(
            any(
                self[s][a] != policy_improvement[s][a]
                for a in policy_improvement[s]
            )
            for s in policy_improvement
        )

        # execute improvement on tabular policy
        self.state_action_prob.update(policy_improvement)

        # check that the action probabilities in each state sum to 1.0
        if not np.allclose(
                [
                    sum(self[s].values())
                    for s in self
                ], 1.0
        ):  # pragma no cover
            raise ValueError('Expected action probabilities to sum to 1.0')

        return num_states_improved

    def __init__(
            self,
            continuous_state_discretization_resolution: Optional[float],
            SS: Optional[List[MdpState]]
    ):
        """
        Initialize the policy.

        :param continuous_state_discretization_resolution: Discretization resolution for continuous state spaces.
        :param SS: List of states for which to initialize the policy to be equiprobable over actions. This is useful for
        environments in which the list of states can be easily enumerated. It is not useful for environments (e.g.,
        `rlai.core.environments.mancala.Mancala`) in which the list of states is very large and difficult enumerate ahead of
        time. The latter problems should be addressed with a lazy-expanding list of states (see Mancala for an example).
        In such cases, pass None here.
        """

        if SS is None:
            SS = []

        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution

        self.state_action_prob: Dict[MdpState, Dict[Action, float]] = {
            s: {
                a: 1 / len(s.AA)
                for a in s.AA
            }
            for s in SS
        }

        self.state_id_str_int: Dict[str, int] = {}

    def __len__(
            self
    ) -> int:
        """
        Get the number of states in the policy.

        :return: Number of states.
        """

        return len(self.state_action_prob)

    def __contains__(
            self,
            state: Optional[MdpState]
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

        return state in self.state_action_prob

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        # if the policy is not defined for the state, then update the policy to be uniform across feasible actions.
        if state not in self.state_action_prob:
            self.state_action_prob[state] = {
                a: 1 / len(state.AA)
                for a in state.AA
            }

        return self.state_action_prob[state]

    def __iter__(
            self
    ) -> Iterator:
        """
        Get an iterator over the policies states and their action-probability dictionaries.
        :return: Iterator.
        """

        return iter(self.state_action_prob)

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current policy equals another.

        :param other: Other policy.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, TabularPolicy):
            raise ValueError(f'Expected {TabularPolicy}')

        return self.state_action_prob == other.state_action_prob

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current policy does not equal another.

        :param other: Other policy.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
