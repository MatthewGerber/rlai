import math
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Union, Optional, Dict, List, Tuple, Any

import numpy as np
from numpy.random import RandomState

from rlai.docs import rl_text
from rlai.utils import get_base_argument_parser, IncrementalSampleAverager, parse_arguments, sample_list_item


@rl_text(chapter='Rewards', page=1)
class Reward:
    """
    Reward.
    """

    def __init__(
            self,
            i: Optional[int],
            r: float
    ):
        """
        Initialize the reward.

        :param i: Identifier for the reward.
        :param r: Reward value.
        """

        self.i = i
        self.r = r

    def __str__(
            self
    ) -> str:
        """
        Get string description of reward.

        :return: String.
        """
        return f'Reward{"" if self.i is None else " " + str(self.i)}: {self.r}'

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for reward.

        :return: Hash code
        """

        assert self.i is not None

        return self.i

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current reward equals another.

        :param other: Other reward.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, Reward):
            raise ValueError(f'Expected {Reward}')

        return self.i == other.i

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current reward is not equal to another.

        :param other: Other reward.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)


@rl_text(chapter='Actions', page=1)
class Action:
    """
    Base class for all actions.
    """

    def __init__(
            self,
            i: int,
            name: Optional[str] = None
    ):
        """
        Initialize the action.

        :param i: Identifier for the action.
        :param name: Name.
        """

        self.i = i
        self.name = name

    def __str__(
            self
    ) -> str:
        """
        Get string description of action.

        :return: String.
        """

        return f'{self.i}:  {"Action" + str(self.i) if self.name is None else self.name}'

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for action.

        :return: Hash code.
        """

        return self.i

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current action equals another.

        :param other: Other action.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, Action):
            raise ValueError(f'Expected {Action}')

        return self.i == other.i

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current action is not equal to another.

        :param other: Other action.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)


@rl_text(chapter='Actions', page=1)
class DiscretizedAction(Action):
    """
    Action that is derived from discretizing an n-dimensional continuous action space.
    """

    def __init__(
            self,
            i: int,
            continuous_value: np.ndarray,
            name: Optional[str] = None
    ):
        """
        Initialize the action.

        :param i: Identifier for the action.
        :param continuous_value: Continuous n-dimensional action.
        :param name: Name (optional).
        """

        super().__init__(
            i=i,
            name=name
        )

        self.continuous_value = continuous_value


@rl_text(chapter=13, page=335)
class ContinuousMultiDimensionalAction(Action):
    """
    Continuous-valued multidimensional action.
    """

    def __init__(
            self,
            value: Optional[np.ndarray],
            min_values: Optional[np.ndarray],
            max_values: Optional[np.ndarray],
            name: Optional[str] = None
    ):
        """
        Initialize the action.

        :param value: Value.
        :param min_values: Minimum values.
        :param max_values: Maximum values.
        :param name: Name.
        """

        super().__init__(
            i=0,
            name=name
        )

        self.value = value
        self.min_values = min_values
        self.max_values = max_values


@rl_text(chapter='States', page=1)
class State:
    """
    Base state for all other states.
    """

    def is_feasible(
            self,
            a: Action
    ) -> bool:
        """
        Check whether an action is feasible from the current state. This uses a set-based lookup with O(1) complexity,
        which is far faster than checking for the action in self.AA.

        :param a: Action.
        :return: True if the action is feasible from the current state and False otherwise.
        """

        return a in self.AA_set

    def __init__(
            self,
            i: Optional[int],
            AA: List[Action]
    ):
        """
        Initialize the state.

        :param i: Identifier for the state.
        :param AA: All actions that can be taken from this state.
        """

        self.i = i
        self.AA = AA

        # use set for fast existence checks (e.g., in `feasible` function)
        self.AA_set = set(self.AA)

    def __str__(
            self
    ) -> str:
        """
        Get string description of state.

        :return: String.
        """
        return f'State {self.i}'

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for state.

        :return: Hash code
        """

        assert self.i is not None

        return self.i

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current state equals another.

        :param other: Other state.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, State):
            raise ValueError(f'Expected {State}')

        return self.i == other.i

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current state is not equal to another.

        :param other: Other state.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)


@rl_text(chapter=3, page=47)
class MdpState(State, ABC):
    """
    State of an MDP.
    """

    def __init__(
            self,
            i: Optional[int],
            AA: List[Action],
            terminal: bool,
            truncated: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions that can be taken from this state.
        :param terminal: Whether the state is terminal, meaning the episode has terminated naturally due to the
        dynamics of the environment. For example, the natural dynamics of the environment might terminate when the agent
        reaches a predefined goal state.
        :param truncated: Whether the state is truncated, meaning the episode has ended for some reason other than the
        natural dynamics of the environment. For example, imposing an artificial time limit on an episode might cause
        the episode to end without the agent in a predefined goal state.
        """

        super().__init__(
            i=i,
            AA=AA
        )

        self.terminal = terminal
        self.truncated = truncated


@rl_text(chapter=1, page=6)
class Policy(ABC):
    """
    Base policy class.
    """

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset the policy for a new run.

        :param state: Initial state.
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
        position within an n-dimensional continuous state space.
        :return: Integer identifier.
        """

        return None

    @abstractmethod
    def __contains__(
            self,
            state: Optional[MdpState]
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

    @abstractmethod
    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """


@rl_text(chapter='Agents', page=1)
class Agent(ABC):
    """
    Base class for all agents.
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
            environment: 'Environment'
    ) -> Tuple[List['Agent'], List[str]]:
        """
        Initialize agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

    def __init__(
            self,
            name: str,
            random_state: RandomState
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        """

        self.name = name
        self.random_state = random_state

        self.most_recent_action: Optional[Action] = None
        self.most_recent_action_tick: Optional[int] = None
        self.most_recent_state: Optional[State] = None
        self.most_recent_state_tick: Optional[int] = None
        self.N_t_A: Dict[Action, float] = {}

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent for a new run.

        :param state: Initial state.
        """

        self.most_recent_action = None
        self.most_recent_action_tick = None
        self.most_recent_state = state
        self.most_recent_state_tick = 0
        self.N_t_A = {a: 0.0 for a in self.N_t_A}

    def sense(
            self,
            state: State,
            t: int
    ):
        """
        Pass the agent state information to sense.

        :param state: State.
        :param t: Time tick for `state`.
        """

        self.most_recent_state = state
        self.most_recent_state_tick = t

    def act(
            self,
            t: int
    ) -> Action:
        """
        Request an action from the agent.

        :param t: Current time step.
        :return: Action
        """

        a = self.__act__(t=t)

        if a is None:
            raise ValueError('Agent returned action of None.')

        assert self.most_recent_state is not None

        if not self.most_recent_state.is_feasible(a):
            raise ValueError(f'Action {a} is not feasible in state {self.most_recent_state}')

        self.most_recent_action = a
        self.most_recent_action_tick = t

        if a not in self.N_t_A:
            self.N_t_A[a] = 0

        self.N_t_A[a] += 1

        return a

    @abstractmethod
    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Request an action from the agent.

        :param t: Current time step.
        :return: Action
        """

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward.
        """

    def __str__(
            self
    ) -> str:
        """
        Return name.

        :return: Name.
        """

        return self.name


@rl_text(chapter=2, page=27)
class QValueAgent(Agent, ABC):
    """
    Nonassociative, q-value agent.
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
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--initial-q-value',
            type=float,
            default=0.0,
            help='Initial Q-value to use for all actions. Use values greater than zero to encourage exploration in the early stages of the run.'
        )

        parser.add_argument(
            '--alpha',
            type=float,
            default=None,
            help='Constant step size for Q-value update. If provided, the Q-value sample average becomes a recency-weighted average (good for nonstationary environments). If `None` is passed, then the unweighted sample average will be used (good for stationary environments).'
        )

        return parser

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            initial_q_value: float,
            alpha: float
    ):
        """
        Initialize the agent.

        :param name: Name of agent.
        :param random_state: Random state.
        :param initial_q_value: Initial Q-value to use for all actions. Use values greater than zero to encourage
        exploration in the early stages of the run.
        :param alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        """

        super().__init__(
            name=name,
            random_state=random_state
        )

        self.initial_q_value = initial_q_value
        self.alpha = alpha

        self.Q: Optional[Dict[Action, IncrementalSampleAverager]] = None

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent to a state prior to any learning.

        :param state: New state.
        """

        super().reset_for_new_run(state)

        assert self.most_recent_state is not None

        if self.Q is None:
            self.Q = {
                a: IncrementalSampleAverager(
                    initial_value=self.initial_q_value,
                    alpha=self.alpha
                )
                for a in self.most_recent_state.AA
            }
        else:
            for averager in self.Q.values():
                averager.reset()

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward value.
        """

        super().reward(r)

        assert self.Q is not None
        assert self.most_recent_action is not None
        self.Q[self.most_recent_action].update(r)


@rl_text(chapter=2, page=27)
class EpsilonGreedyQValueAgent(QValueAgent):
    """
    Nonassociative, epsilon-greedy agent.
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
            '--epsilon',
            type=float,
            nargs='+',
            help='Space-separated list of epsilon values to evaluate.'
        )

        parser.add_argument(
            '--epsilon-reduction-rate',
            type=float,
            default=0.0,
            help='Percentage reduction of epsilon from its initial value. This is applied at each time step when the agent explores. For example, pass 0 for no reduction and 0.01 for a 1-percent reduction at each exploration step.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: 'Environment'
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # grab and delete epsilons from parsed arguments
        epsilons = parsed_args.epsilon
        del parsed_args.epsilon

        # initialize agents
        agents: List[Agent] = [
            cls(
                name=f'epsilon-greedy (e={epsilon:0.2f})',
                random_state=random_state,
                epsilon=epsilon,
                **vars(parsed_args)
            )
            for epsilon in epsilons
        ]

        return agents, unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            initial_q_value: float,
            alpha: float,
            epsilon: float,
            epsilon_reduction_rate: float
    ):
        """
        Initialize the agent.

        :param name: Name of agent.
        :param random_state: Random state.
        :param initial_q_value: Initial Q-value to use for all actions. Use values greater than zero to encourage
        exploration in the early stages of the run.
        :param alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        :param epsilon: Epsilon.
        :param epsilon_reduction_rate: Rate at which `epsilon` is reduced from its initial value to zero. This is the
        percentage reduction, and it is applied at each time step when the agent explores. For example, pass 0 for no
        reduction and 0.01 for a 1-percent reduction at each exploration step.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            initial_q_value=initial_q_value,
            alpha=alpha
        )

        self.epsilon = self.original_epsilon = epsilon
        self.epsilon_reduction_rate = epsilon_reduction_rate
        self.greedy_action: Optional[Action] = None

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent to a state prior to any learning.

        :param state: New state.
        """

        super().reset_for_new_run(state)

        self.epsilon = self.original_epsilon
        assert self.Q is not None
        self.greedy_action = list(self.Q.keys())[0]

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Act in an epsilon-greedy fashion.

        :param t: Current time step.
        :return: Action.
        """

        if self.random_state.random_sample() < self.epsilon:
            a: Action = self.random_state.choice(self.most_recent_state.AA)  # type: ignore
            self.epsilon *= (1 - self.epsilon_reduction_rate)
        else:
            assert self.greedy_action is not None
            a = self.greedy_action

        assert isinstance(a, Action)

        return a

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward value.
        """

        super().reward(r)

        Q = self.Q
        assert Q is not None

        # get new greedy action, which might have changed
        self.greedy_action = max(Q.items(), key=lambda action_value: action_value[1].get_value())[0]


@rl_text(chapter=2, page=35)
class UpperConfidenceBoundAgent(QValueAgent):
    """
    Nonassociative, upper-confidence-bound agent.
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
            '--c',
            type=float,
            nargs='+',
            help='Space-separated list of confidence levels (higher gives more exploration).'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: 'Environment'
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # grab and delete c values from parsed arguments
        c_values = parsed_args.c
        del parsed_args.c

        # initialize agents
        agents: List[Agent] = [
            cls(
                name=f'UCB (c={c})',
                random_state=random_state,
                c=c,
                **vars(parsed_args)
            )
            for c in c_values
        ]

        return agents, unparsed_args

    def get_denominator(
            self,
            a: Action,
    ) -> float:
        """
        Get denominator of UCB action rule.

        :param a: Action.
        :return: Denominator.
        """

        if a not in self.N_t_A or self.N_t_A[a] == 0:
            return sys.float_info.min
        else:
            return self.N_t_A[a]

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Act according to the upper-confidence-bound rule. This gives the benefit of the doubt to actions that have not
        been selected as frequently as other actions, that their values will be good.

        :param t: Current time step.
        :return: Action.
        """

        assert self.most_recent_state is not None
        Q = self.Q
        assert Q is not None
        return max(
            self.most_recent_state.AA,
            key=lambda a: Q[a].get_value() + self.c * math.sqrt(math.log(t + 1) / self.get_denominator(a))
        )

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            initial_q_value: float,
            alpha: float,
            c: float
    ):
        """
        Initialize the agent.

        :param name: Name of agent.
        :param random_state: Random state.
        :param initial_q_value: Initial Q-value to use for all actions. Use values greater than zero to encourage
        exploration in the early stages of the run.
        :param alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        :param c: Confidence.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            initial_q_value=initial_q_value,
            alpha=alpha
        )

        self.c = c


@rl_text(chapter=2, page=37)
class PreferenceGradientAgent(Agent):
    """
    Preference-gradient agent.
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
            '--step-size-alpha',
            type=float,
            default=0.1,
            help='Step-size parameter used to update action preferences.'
        )

        parser.add_argument(
            '--reward-average-alpha',
            type=float,
            default=None,
            help='Constant step-size for reward averaging. If provided, the reward average becomes a recency-weighted average (good for nonstationary environments). If `None` is passed, then the unweighted sample average will be used (good for stationary environments).'
        )

        parser.add_argument(
            '--use-reward-baseline',
            action='store_true',
            help='Pass this flag to use a reward baseline when updating action preferences.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: 'Environment'
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # initialize agents
        agents: List[Agent] = [
            cls(
                name=f'preference gradient (step size={parsed_args.step_size_alpha})',
                random_state=random_state,
                **vars(parsed_args)
            )
        ]

        return agents, unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            step_size_alpha: float,
            use_reward_baseline: bool,
            reward_average_alpha: float
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random State.
        :param step_size_alpha: Step-size parameter used to update action preferences.
        :param use_reward_baseline: Whether to use a reward baseline when updating action preferences.
        :param reward_average_alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        """

        super().__init__(
            name=name,
            random_state=random_state
        )

        self.step_size_alpha = step_size_alpha
        self.use_reward_baseline = use_reward_baseline
        self.R_bar = IncrementalSampleAverager(
            initial_value=0.0,
            alpha=reward_average_alpha
        )

        self.H_t_A: Optional[np.ndarray] = None
        self.Pr_A: Optional[np.ndarray] = None

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent to a state prior to any learning.

        :param state: New state.
        """

        super().reset_for_new_run(state)

        assert self.most_recent_state is not None
        self.H_t_A = np.zeros(len(self.most_recent_state.AA))
        self.update_action_probabilities()
        self.R_bar.reset()

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Sample a random action based on current preferences.

        :param t: Time step.
        :return: Action.
        """

        assert self.most_recent_state is not None
        return sample_list_item(self.most_recent_state.AA, self.Pr_A, self.random_state)

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward value.
        """

        super().reward(r)

        if self.use_reward_baseline:
            self.R_bar.update(r)
            preference_update = self.step_size_alpha * (r - self.R_bar.get_value())
        else:
            preference_update = self.step_size_alpha * r

        assert self.H_t_A is not None
        assert self.Pr_A is not None

        # get preference update for action taken
        assert self.most_recent_action is not None
        most_recent_action_i = self.most_recent_action.i
        update_action_taken = self.H_t_A[most_recent_action_i] + preference_update * (1 - self.Pr_A[most_recent_action_i])

        # get other-action preference update for all actions
        update_all = self.H_t_A - preference_update * self.Pr_A

        # set preferences
        self.H_t_A = update_all
        assert self.H_t_A is not None
        self.H_t_A[most_recent_action_i] = update_action_taken

        self.update_action_probabilities()

    def update_action_probabilities(
            self
    ):
        """
        Update action probabilities based on current preferences.
        """

        assert self.H_t_A is not None

        exp_h_t_a = np.e ** self.H_t_A
        exp_h_t_a_sum = exp_h_t_a.sum()

        self.Pr_A = exp_h_t_a / exp_h_t_a_sum


@rl_text(chapter='Agents', page=1)
class MdpAgent(Agent, ABC):
    """
    MDP agent. Adds the concepts of state, reward discounting, and policy-based action to the base agent.
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
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--gamma',
            type=float,
            help='Discount factor.'
        )

        return parser

    def shape_reward(
            self,
            reward: Reward,
            first_t: int,
            final_t: int
    ) -> Dict[int, float]:
        """
        Shape a reward value that has been obtained. Reward shaping entails the calculation of time steps at which
        returns should be updated along with the weighted reward for each. This function applies exponential discounting
        based on the value of gamma specified in the current agent (i.e., the traditional reward shaping approach
        discussed by Sutton and Barto). Subclasses are free to override the current function and shape rewards as needed
        for the task at hand.

        :param reward: Obtained reward.
        :param first_t: First time step at which to shape reward value.
        :param final_t: Final time step at which to shape reward value.
        :return: Dictionary of time steps and associated shaped rewards.
        """

        # shape reward from the first time step through the final time step, including both endpoints.
        t_shaped_reward = {
            t: self.gamma ** (final_t - t) * reward.r
            for t in range(first_t, final_t + 1)
        }

        return t_shaped_reward

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            pi: Policy,
            gamma: float
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param: Policy.
        :param gamma: Discount.
        """

        super().__init__(
            name=name,
            random_state=random_state
        )

        self.pi = pi
        self.gamma = gamma


@rl_text(chapter='Agents', page=1)
class StochasticMdpAgent(MdpAgent, ABC):
    """
    Stochastic MDP agent. Adds random selection of actions based on probabilities specified in the agent's policy.
    """

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent for a new run.

        :param state: Initial state.
        """

        super().reset_for_new_run(state)

        assert isinstance(state, MdpState)
        self.pi.reset_for_new_run(state)

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Act stochastically according to the policy.

        :param t: Time tick.
        :return: Action.
        """

        self.most_recent_state: MdpState

        # sample action according to policy for most recent state
        action_prob = self.pi[self.most_recent_state]
        actions = list(action_prob.keys())
        probs = np.array(list(action_prob.values()))

        return sample_list_item(
            x=actions,
            probs=probs,
            random_state=self.random_state
        )

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            pi: Policy,
            gamma: float
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param pi: Policy.
        :param gamma: Discount.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            pi=pi,
            gamma=gamma
        )


@rl_text(chapter='Agents', page=1)
class Human(Agent):
    """
    An interactive, human-driven agent that prompts for actions at each time step.
    """

    class DummyPolicy(Policy):
        """
        A dummy policy to make the present agent compatible with our runners.
        """

        def __contains__(
                self,
                state: Optional[MdpState]
        ) -> bool:
            """
            Check whether the policy is defined for a state.

            :param state: State.
            :return: True if policy is defined for state and False otherwise.
            """

            return True

        def __getitem__(
                self,
                state: Optional[MdpState]
        ) -> Dict[Action, float]:
            """
            Check whether the policy is defined for a state.

            :param state: State.
            :return: True if policy is defined for state and False otherwise.
            """

            return {}

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: 'Environment'
    ) -> Tuple[List[Agent], List[str]]:
        """
        Not implemented.
        """

        raise NotImplementedError()

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Prompt the human user for input.

        :param t: Time step.
        :return: Action.
        """

        action = None

        while action is None:

            prompt = 'Please select from the following actions:  '

            self.most_recent_state: MdpState

            a_name_i = {
                a.name: i
                for i, a in enumerate(self.most_recent_state.AA)
            }

            for i, name in enumerate(sorted(a_name_i.keys())):  # type: ignore[type-var]
                prompt += f'{", " if i > 0 else ""}{name}'

            prompt += '\nEnter your selection:  '

            # noinspection PyBroadException
            try:
                chosen_name = self.get_input(prompt)
                action = self.most_recent_state.AA[a_name_i[chosen_name]]
            except Exception:
                pass

        return action

    @staticmethod
    def get_input(
            prompt: str
    ) -> str:
        """
        Get input from the human agent.

        :param prompt: Prompt.
        :return: Input.
        """

        # don't compute coverage for the following. we mock the current function for tests, but the input function
        # can't be patched. https://stackoverflow.com/questions/21046717/python-mocking-raw-input-in-unittests
        return input(prompt)  # pragma no cover

    def __init__(
            self
    ):
        """
        Initialize the agent.
        """

        super().__init__(
            name='human',
            random_state=RandomState(12345)
        )

        self.pi = Human.DummyPolicy()


class Environment(ABC):
    """
    Base class for all environments.
    """

    @classmethod
    def get_argument_parser(
            cls,
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = get_base_argument_parser()

        parser.add_argument(
            '--T',
            type=int,
            help='Maximum number of time steps to run.'
        )

        return parser

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple['Environment', List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

    @abstractmethod
    def reset_for_new_run(
            self,
            agent: Any
    ) -> Optional[State]:
        """
        Reset the environment.

        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: New state.
        """

        self.num_resets += 1

        return None

    def run(
            self,
            agent: Any,
            monitor: 'Monitor'
    ):
        """
        Run the environment with an agent. This routine does not provide any learning functionality. It only steps
        through the environment with the agent.

        :param agent: Agent to run.
        :param monitor: Monitor.
        """

        t = 0
        while (self.T is None or t < self.T) and not self.run_step(t, agent, monitor):
            t += 1

    @abstractmethod
    def run_step(
            self,
            t: int,
            agent: Any,
            monitor: 'Monitor'
    ) -> bool:
        """
        Run a step of the environment with an agent.

        :param t: Step.
        :param agent: Agent.
        :param monitor: Monitor.
        :return: True if a terminal state was entered and the run should terminate, and False otherwise.
        """

    def exiting_episode_without_termination(
            self
    ):
        """
        Called when a learning procedure is exiting the episode without natural termination (e.g., after truncation).
        The episode will not reach a natural termination state. Instead, the episode loop will exit. This function is
        called to provide the environment an opportunity to clean up resources. This is not usually needed with
        simulation-based environments since breaking the episode loop prevents any further episode advancement. However,
        in physical environments the system might continue to advance in the absence of further calls to the advance
        function. This function allows the environment to perform any adjustments that are normally required upon
        termination.
        """

    def close(
            self
    ):
        """
        Close the environment, releasing resources.
        """

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int]
    ):
        """
        Initialize the environment.

        :param name: Name of the environment.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        """

        self.name = name
        self.random_state = random_state
        self.T = T

        self.num_resets = 0

    def __str__(
            self
    ) -> str:
        """
        Return name.

        :return: Name.
        """

        return self.name


class Monitor:
    """
    Monitor for runs of an environment with an agent.
    """

    def __init__(
            self
    ):
        """
        Initialize the monitor.
        """

        self.t_count_optimal_action = {}
        self.t_average_reward = {}
        self.t_average_cumulative_reward = {}
        self.cumulative_reward = 0.0
        self.most_recent_time_step: Optional[int] = None

    def reset_for_new_run(
            self
    ):
        """
        Reset the monitor for a new run.
        """

        self.cumulative_reward = 0.0

    def report(
            self,
            t: int,
            agent_action: Optional[Action] = None,
            optimal_action: Optional[Action] = None,
            action_reward: Optional[float] = None
    ):
        """
        Report information about a run.

        :param t: Time step.
        :param agent_action: Action taken.
        :param optimal_action: Optimal action.
        :param action_reward: Reward obtained.
        """

        if t not in self.t_count_optimal_action:
            self.t_count_optimal_action[t] = 0

        if t not in self.t_average_reward:
            self.t_average_reward[t] = IncrementalSampleAverager()

        if t not in self.t_average_cumulative_reward:
            self.t_average_cumulative_reward[t] = IncrementalSampleAverager()

        if agent_action is not None and optimal_action is not None and agent_action == optimal_action:
            self.t_count_optimal_action[t] += 1

        if action_reward is not None:
            self.t_average_reward[t].update(action_reward)
            self.cumulative_reward += action_reward
            self.t_average_cumulative_reward[t].update(self.cumulative_reward)

        self.most_recent_time_step = t
