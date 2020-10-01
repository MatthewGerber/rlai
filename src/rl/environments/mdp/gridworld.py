from argparse import ArgumentParser, Namespace
from typing import List, Tuple, Union, Optional

import numpy as np
from numpy.random import RandomState

from rl.actions import Action
from rl.agents import Agent
from rl.environments import Environment
from rl.environments.mdp import MDP
from rl.meta import rl_text
from rl.rewards import Reward
from rl.runners.monitor import Monitor
from rl.states.mdp import MdpState


@rl_text(chapter=3, page=60)
class Gridworld(MDP):

    @staticmethod
    def example_4_1(
    ) -> Environment:

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

        g.set_down_p(
            r=r_minus_one,
            value=1.0
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

    def set_p(
            self,
            states: np.array,
            a: Optional[Union[Action, List[Action]]],
            s: Optional[Union[MdpState, List[MdpState]]],
            r: Optional[Union[Reward, List[Reward]]],
            value: float
    ):
        if a is None:
            a = self.AA

        if not isinstance(a, list):
            a = [a]

        if s is None:
            s = self.SS

        if not isinstance(s, list):
            s = [s]

        if r is None:
            r = self.RR

        if not isinstance(r, list):
            r = [r]

        state: MdpState
        for state in states:
            for a in a:
                for s in s:
                    for r in r:
                        state.p_S_prime_R_given_A[a][s][r] = value

    def set_down_p(
            self,
            r: Reward,
            value: float
    ):
        grid = self.grid
        for row_i, next_row_i in zip(range(grid.shape[0]), range(1, grid.shape[0])):
            self.set_p(
                states=grid[row_i:],
                a=self.a_down,
                s=grid[next_row_i:],
                r=r,
                value=value
            )

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            n_rows: int,
            n_columns: int,
            RR: List[Reward]
    ):
        AA = [
            Action(i=a)
            for a in ['u', 'd', 'l', 'r']
        ]

        self.a_up, self.a_down, self.a_left, self.a_right = AA

        super().__init__(
            name=name,
            AA=AA,
            random_state=random_state,
            SS=[
                MdpState(
                    i=row_i * n_columns + col_j,
                    AA=self.AA,
                    SS=self.SS,
                    RR=self.RR
                )
                for row_i in range(n_rows)
                for col_j in range(n_columns)
            ],
            RR=RR
        )

        self.grid = np.array(self.SS).reshape(n_rows, n_columns)
