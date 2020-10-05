from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Tuple

import numpy as np
from numpy.random import RandomState

from rl.actions import Action
from rl.agents import Agent
from rl.environments.mdp import MdpEnvironment
from rl.states.mdp import MdpState
from rl.utils import sample_list_item


class MdpAgent(Agent, ABC):
    """
    MDP agent.
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
            default=0.1,
            help='Discount factor.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args
    
    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            SS: List[MdpState],
            gamma: float
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions, with identifiers sorted in increasing order from zero.
        :param name: Name of the agent.
        :param random_state: Random state.
        :param SS: List of all possible states, with identifiers sorted in increasing order from zero.
        :param gamma: Discount.
        """

        for i, s in enumerate(SS):
            if s.i != i:
                raise ValueError('States must be sorted in increasing order from zero.')

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state
        )

        self.SS = SS
        self.gamma = gamma

        self.pi = {
            s: {
                a: np.nan
                for a in self.AA
            }
            for s in self.SS
        }


class Stochastic(MdpAgent):
    """
    Stochastic MDP agent.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment,
            random_state: RandomState
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :param random_state: Random state.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        agents = [
            Stochastic(
                AA=environment.AA,
                name=f'stochastic (gamma={parsed_args.gamma})',
                random_state=random_state,
                SS=environment.SS,
                gamma=parsed_args.gamma
            )
        ]

        return agents, unparsed_args

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Act randomly.

        :param t: Time tick.
        :return: Action.
        """

        return sample_list_item(self.AA, np.array([self.pi[self.most_recent_state][a] for a in self.AA]), self.random_state)

    def reward(self, r: float):
        pass

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            SS: List[MdpState],
            gamma: float
    ):
        """
        Initialize the agent with an equiprobable policy over actions.

        :param AA: List of all possible actions, with identifiers sorted in increasing order from zero.
        :param name: Name of the agent.
        :param random_state: Random state.
        :param SS: List of all possible states, with identifiers sorted in increasing order from zero.
        :param gamma: Discount.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state,
            SS=SS,
            gamma=gamma
        )

        num_actions = len(self.AA)
        
        self.pi = {
            s: {
                a: 1 / num_actions
                for a in self.AA
            }
            for s in self.SS
        }
