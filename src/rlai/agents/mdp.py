from abc import ABC
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.utils import sample_list_item, parse_arguments


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
            curr_t: int,
            reward: Reward,
            n_steps: Optional[int],
            t_state_a_g: Dict[int, Tuple[MdpState, Action, float]]
    ) -> List[Tuple[int, float]]:
        """
        Shape a reward value that has been obtained. Reward shaping entails the calculation of time steps at which
        returns should be updated along with weight associated with each. This function applies exponential discounting
        based on the value of gamma specified in the current agent (the traditional shaping approach discussed by Sutton
        and Barto). Subclasses are free to override the current function and shape rewards as needed for the task at
        hand.

        :param curr_t: Current time step.
        :param reward: Reward obtained from the action taken at the current time step.
        :param n_steps: Number of steps to use in n-step (bootstrapped) returns, or None for Monte Carlo returns (i.e.,
        infinite steps and no bootstrapping).
        :param t_state_a_g: Truncated return accumulator.
        :return: List of time steps for which returns should be updated, along with weights.
        """

        # if n_steps is None, then get all prior time steps (equivalent to infinite n_steps, or monte carlo returns).
        if n_steps is None:
            return_t_weight = [
                (t, self.gamma ** (curr_t - t))
                for t in t_state_a_g.keys()
            ]
        else:
            # in 1-step td, the earliest time step is the current time step; in 2-step, the earliest time step is the
            # prior time step, etc. always update returns from the earliest time step through the current time step,
            # inclusive.
            earliest_t = max(0, curr_t - n_steps + 1)
            return_t_weight = [
                (t, self.gamma ** (curr_t - t))
                for t in range(earliest_t, curr_t + 1)
            ]

        return return_t_weight

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
class StochasticMdpAgent(MdpAgent):
    """
    Stochastic MDP agent. Adds random selection of actions based on probabilities specified in the agent's policy.
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

        agents = [
            StochasticMdpAgent(
                name=f'stochastic (gamma={parsed_args.gamma})',
                random_state=random_state,
                pi=pi,
                **vars(parsed_args)
            )
        ]

        return agents, unparsed_args

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
