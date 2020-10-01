from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Tuple

import numpy as np
from numpy.random import RandomState

from rl.actions import Action
from rl.agents import Agent
from rl.environments.mdp import MDP
from rl.states import State
from rl.utils import sample_list_item


class PolicyAgent(Agent, ABC):

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            SS: List[State],
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


class Equipropable(PolicyAgent):

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

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MDP,
            random_state: RandomState
    ) -> Tuple[List[Agent], List[str]]:

        parsed_args, unparsed_args = cls.parse_arguments(args)

        agents = [
            Equipropable(
                AA=environment.AA,
                name='equiprobable',
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

        return sample_list_item(self.AA, self.pi, self.random_state)

    def reward(self, r: float):
        pass

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            SS: List[State],
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
