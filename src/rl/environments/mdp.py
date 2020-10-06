from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Tuple

import numpy as np
from numpy.random import RandomState

from rl.actions import Action
from rl.agents import Agent
from rl.environments import Environment
from rl.meta import rl_text
from rl.rewards import Reward
from rl.runners.monitor import Monitor
from rl.states.mdp import MdpState


@rl_text(chapter=3, page=47)
class MdpEnvironment(Environment, ABC):
    """
    MDP environment.
    """

    def __init__(
            self,
            name: str,
            AA: List[Action],
            random_state: RandomState,
            SS: List[MdpState],
            RR: List[Reward]
    ):
        super().__init__(
            name=name,
            AA=AA,
            random_state=random_state
        )

        self.SS = SS
        self.RR = RR


@rl_text(chapter=3, page=60)
class Gridworld(MdpEnvironment):
    """
    Gridworld MDP environment.
    """

    @staticmethod
    def example_4_1(
    ):
        """
        Construct the Gridworld for Example 4.1.
        :return: Gridworld.
        """

        RR = [
            Reward(
                i=i,
                r=r
            )
            for i, r in enumerate([0, -1])
        ]

        r_zero, r_minus_one = RR

        g = Gridworld(
            name='Example 4.1',
            random_state=None,
            n_rows=4,
            n_columns=4,
            RR=RR
        )

        g.grid[0, 0].terminal = g.grid[3, 3].terminal = True

        g.set_model_probabilities(
            nonterminal_reward=r_minus_one,
            terminal_reward=r_zero
        )

        return g

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
            '--id',
            type=str,
            default='example_4_1',
            help='Gridworld identifier.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        gridworld = getattr(cls, parsed_args.id)()

        return gridworld, unparsed_args

    def run(
            self,
            agent: Agent,
            T: int,
            monitor: Monitor
    ):
        pass

    def set_model_probabilities(
            self,
            nonterminal_reward: Reward,
            terminal_reward: Reward
    ):
        """
        Set model probabilities within the environment.

        :param nonterminal_reward: Nonterminal reward.
        :param terminal_reward: Terminal reward.
        """

        # set nonterminal reward probabilities
        for a in self.AA:

            # arrange grid such that a row-by-row will generate the appropriate state transition sequences
            if a == self.a_down:
                grid = self.grid
            elif a == self.a_up:
                grid = np.flipud(self.grid)
            elif a == self.a_right:
                grid = self.grid.transpose()
            elif a == self.a_left:
                grid = np.flipud(self.grid.transpose())
            else:
                raise ValueError(f'Unknown action:  {a}')

            # go row by row, with the final row transitioning to itself
            for s_row_i, s_prime_row_i in zip(range(grid.shape[0]), list(range(1, grid.shape[0])) + [-1]):
                for s, s_prime in zip(grid[s_row_i, :], grid[s_prime_row_i, :]):
                    if not s.terminal:
                        s.p_S_prime_R_given_A[a][s_prime][nonterminal_reward] = 1.0

        # set terminal reward probabilities
        s: MdpState
        for s in self.SS:
            if s.terminal:
                for a in self.AA:
                    s.p_S_prime_R_given_A[a][s][terminal_reward] = 1.0

        # check that marginal probabilities for each state sum to 1
        for s in self.SS:
            for a in self.AA:

                marginal_prob = sum([
                    s.p_S_prime_R_given_A[a][s_prime][r]
                    for s_prime in s.p_S_prime_R_given_A[a]
                    for r in s.p_S_prime_R_given_A[a][s_prime]
                ])

                if marginal_prob != 1.0:
                    raise ValueError(f'Expected state-marginal probability of 1.0, got {marginal_prob}.')

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            n_rows: int,
            n_columns: int,
            RR: List[Reward]
    ):
        """
        Initialize the gridworld.

        :param name: Name.
        :param random_state: Random state.
        :param n_rows: Number of row.
        :param n_columns: Number of columns.
        :param RR: List of all possible rewards.
        """

        AA = [
            Action(
                i=i,
                name=direction
            )
            for i, direction in enumerate(['u', 'd', 'l', 'r'])
        ]

        self.a_up, self.a_down, self.a_left, self.a_right = AA

        # create mdp states, providing each with the list of all states
        SS = []
        SS.extend([
            MdpState(
                i=row_i * n_columns + col_j,
                AA=AA,
                SS=SS,
                RR=RR,
                terminal=False
            )
            for row_i in range(n_rows)
            for col_j in range(n_columns)
        ])

        # initialize the model within each state, now that SS has been populated.
        for s in SS:
            s.init_model()

        super().__init__(
            name=name,
            AA=AA,
            random_state=random_state,
            SS=SS,
            RR=RR
        )

        self.grid = np.array(self.SS).reshape(n_rows, n_columns)
