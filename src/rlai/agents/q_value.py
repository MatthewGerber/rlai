import math
import sys
from abc import ABC
from argparse import ArgumentParser
from typing import List, Dict, Tuple, Optional

from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states import State
from rlai.utils import IncrementalSampleAverager, parse_arguments


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
            pi: Optional[Policy]
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param pi: Policy.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # grab and delete epsilons from parsed arguments
        epsilons = parsed_args.epsilon
        del parsed_args.epsilon

        # initialize agents
        agents = [
            EpsilonGreedy(
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
            pi: Optional[Policy]
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param pi: Policy.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # grab and delete c values from parsed arguments
        c_values = parsed_args.c
        del parsed_args.c

        # initialize agents
        agents = [
            UpperConfidenceBound(
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
