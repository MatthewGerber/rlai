import math
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict

import numpy as np
from numpy.random import RandomState

from rlai.core.actions import Action
from rlai.core.environments import Environment
from rlai.core.policies import Policy
from rlai.core.rewards import Reward
from rlai.core.states import State, MdpState
from rlai.meta import rl_text
from rlai.policy_gradient.policies import ParameterizedPolicy
from rlai.state_value import StateValueEstimator
from rlai.utils import (
    get_base_argument_parser,
    parse_arguments,
    sample_list_item,
    IncrementalSampleAverager,
    load_class
)


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
            environment: Environment
    ) -> Tuple[List['Agent'], List[str]]:
        """
        Initialize a agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

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
        self.N_t_A: Dict[Action, int] = {}

    def __str__(
            self
    ) -> str:
        """
        Return name.

        :return: Name.
        """

        return self.name


@rl_text(chapter=2, page=27)
class QValue(Agent, ABC):
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

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent to a state prior to any learning.

        :param state: New state.
        """

        super().reset_for_new_run(state)

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

        self.Q[self.most_recent_action].update(r)

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


@rl_text(chapter=2, page=27)
class EpsilonGreedy(QValue):
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
            environment: Environment
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
        agents = [
            cls(
                name=f'epsilon-greedy (e={epsilon:0.2f})',
                random_state=random_state,
                epsilon=epsilon,
                **vars(parsed_args)
            )
            for epsilon in epsilons
        ]

        return agents, unparsed_args

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
            a = self.random_state.choice(self.most_recent_state.AA)
            self.epsilon *= (1 - self.epsilon_reduction_rate)
        else:
            a = self.greedy_action

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

        # get new greedy action, which might have changed
        self.greedy_action = max(self.Q.items(), key=lambda action_value: action_value[1].get_value())[0]

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
        self.greedy_action = None


@rl_text(chapter=2, page=35)
class UpperConfidenceBound(QValue):
    """
    Nonassociatve, upper-confidence-bound agent.
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
            environment: Environment
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
        agents = [
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

        return max(self.most_recent_state.AA, key=lambda a: self.Q[a].get_value() + self.c * math.sqrt(math.log(t + 1) / self.get_denominator(a)))

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
class PreferenceGradient(Agent):
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
            environment: Environment
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
        agents = [
            cls(
                name=f'preference gradient (step size={parsed_args.step_size_alpha})',
                random_state=random_state,
                **vars(parsed_args)
            )
        ]

        return agents, unparsed_args

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent to a state prior to any learning.

        :param state: New state.
        """

        super().reset_for_new_run(state)

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

        # get preference update for action taken
        most_recent_action_i = self.most_recent_action.i
        update_action_taken = self.H_t_A[most_recent_action_i] + preference_update * (1 - self.Pr_A[most_recent_action_i])

        # get other-action preference update for all actions
        update_all = self.H_t_A - preference_update * self.Pr_A

        # set preferences
        self.H_t_A = update_all
        self.H_t_A[most_recent_action_i] = update_action_taken

        self.update_action_probabilities()

    def update_action_probabilities(
            self
    ):
        """
        Update action probabilities based on current preferences.
        """

        exp_h_t_a = np.e ** self.H_t_A
        exp_h_t_a_sum = exp_h_t_a.sum()

        self.Pr_A = exp_h_t_a / exp_h_t_a_sum

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
        :param use_reward_baseline: Whether or not to use a reward baseline when updating action preferences.
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

        self.H_t_A: Optional[np.array] = None
        self.Pr_A: Optional[np.array] = None


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
            state: MdpState
    ):
        """
        Reset the agent for a new run.

        :param state: Initial state.
        """

        super().reset_for_new_run(state)

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
class ActionValueMdpAgent(StochasticMdpAgent):
    """
    A stochastic MDP agent whose policy is updated according to action-value estimates.
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
            q_S_A
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


@rl_text(chapter='Agents', page=1)
class ParameterizedMdpAgent(StochasticMdpAgent):
    """
    A stochastic MDP agent whose policy is updated according to its parameterized gradients.
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
            '--policy',
            type=str,
            help='Fully-qualified type name of policy to use.'
        )

        parser.add_argument(
            '--v-S',
            type=str,
            help='Fully-qualified type name of baseline state-value estimator to use, or ignore for no baseline.'
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

        # load state-value estimator, which is optional.
        v_S = None
        if parsed_args.v_S is not None:
            estimator_class = load_class(parsed_args.v_S)
            v_S, unparsed_args = estimator_class.init_from_arguments(
                args=unparsed_args,
                random_state=random_state,
                environment=environment
            )
        del parsed_args.v_S

        # load parameterized policy
        policy_class = load_class(parsed_args.policy)
        policy, unparsed_args = policy_class.init_from_arguments(
            args=unparsed_args,
            environment=environment
        )
        del parsed_args.policy

        # noinspection PyUnboundLocalVariable
        agent = cls(
            name=f'parameterized (gamma={parsed_args.gamma})',
            random_state=random_state,
            pi=policy,
            v_S=v_S,
            **vars(parsed_args)
        )

        return [agent], unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            pi: ParameterizedPolicy,
            gamma: float,
            v_S: Optional[StateValueEstimator]
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param pi: Policy.
        :param gamma: Discount.
        :param v_S: Baseline state-value estimator.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            pi=pi,
            gamma=gamma
        )

        self.v_S = v_S


@rl_text(chapter='Agents', page=1)
class Human(Agent):
    """
    An interactive, human-driven agent that prompts for actions at each time step.
    """

    class DummyPolicy(Policy):

        def __contains__(self, state: MdpState) -> bool:
            return True

        def __getitem__(self, state: MdpState) -> Dict[Action, float]:
            return {}

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: Environment
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

            for i, name in enumerate(sorted(a_name_i.keys())):
                prompt += f'{", " if i > 0 else ""}{name}'

            prompt += '\nEnter your selection:  '

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
