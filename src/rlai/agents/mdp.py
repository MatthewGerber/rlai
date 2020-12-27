from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Tuple, Optional

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState
from rlai.utils import sample_list_item


@rl_text(chapter='Agents', page=1)
class MdpAgent(Agent, ABC):
    """
    MDP agent. Adds the concepts of state, reward discounting, and policy-based action to the base agent.
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
            '--gamma',
            type=float,
            help='Discount factor.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

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

        parsed_args, unparsed_args = cls.parse_arguments(args)

        agents = [
            StochasticMdpAgent(
                name=f'stochastic (gamma={parsed_args.gamma})',
                random_state=random_state,
                pi=pi,
                **vars(parsed_args)
            )
        ]

        return agents, unparsed_args

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
