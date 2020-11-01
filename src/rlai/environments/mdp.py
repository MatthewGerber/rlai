from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Tuple, Optional, final

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.environments import Environment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.runners.monitor import Monitor
from rlai.states import State
from rlai.states.mdp import MdpState, ModelBasedMdpState


@rl_text(chapter=3, page=47)
class MdpEnvironment(Environment, ABC):
    """
    MDP environment.
    """

    def reset_for_new_run(
            self
    ) -> State:
        """
        Reset the the environment.
        """

        self.state = self.random_state.choice(self.nonterminal_states)

        return self.state

    @final
    def run_step(
            self,
            t: int,
            agent: Agent,
            monitor: Monitor
    ) -> bool:
        """
        Run a step of the environment with an agent.

        :param t: Step.
        :param agent: Agent.
        :param monitor: Monitor.
        :return: True if a terminal state was entered and the run should terminate, and False otherwise.
        """

        a = agent.act(t=t)

        self.state, next_t, next_reward = self.state.advance(
            environment=self,
            t=t,
            a=a
        )

        agent.sense(
            state=self.state,
            t=next_t
        )

        agent.reward(next_reward.r)
        monitor.report(t=next_t, action_reward=next_reward.r)

        return self.state.terminal

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            SS: List[MdpState],
            RR: List[Reward]
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param SS: List of states.
        :param RR: List of rewards.
        """

        super().__init__(
            name=name,
            random_state=random_state
        )

        self.SS = SS
        self.RR = RR
        self.terminal_states = [s for s in self.SS if s.terminal]
        self.nonterminal_states = [s for s in self.SS if not s.terminal]
        self.state: Optional[MdpState] = None


@rl_text(chapter=3, page=60)
class Gridworld(MdpEnvironment):
    """
    Gridworld MDP environment.
    """

    @staticmethod
    def example_4_1(
            random_state: RandomState
    ):
        """
        Construct the Gridworld for Example 4.1.

        :param random_state: Random state.
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
            random_state=random_state,
            n_rows=4,
            n_columns=4,
            terminal_states=[(0, 0), (3, 3)],
            RR=RR
        )

        # set nonterminal reward probabilities
        for a in [g.a_up, g.a_down, g.a_left, g.a_right]:

            # arrange grid such that a row-to-row scan will generate the appropriate state transition sequences for the
            # current action.
            if a == g.a_down:
                grid = g.grid
            elif a == g.a_up:
                grid = np.flipud(g.grid)
            elif a == g.a_right:
                grid = g.grid.transpose()
            elif a == g.a_left:
                grid = np.flipud(g.grid.transpose())
            else:
                raise ValueError(f'Unknown action:  {a}')

            # go row by row, with the final row transitioning to itself
            for s_row_i, s_prime_row_i in zip(range(grid.shape[0]), list(range(1, grid.shape[0])) + [-1]):
                for s, s_prime in zip(grid[s_row_i, :], grid[s_prime_row_i, :]):
                    if not s.terminal:
                        s.p_S_prime_R_given_A[a][s_prime][r_minus_one] = 1.0

        # set terminal reward probabilities
        s: ModelBasedMdpState
        for s in g.SS:
            if s.terminal:
                for a in s.AA:
                    s.p_S_prime_R_given_A[a][s][r_zero] = 1.0

        for s in g.SS:
            s.check_marginal_probabilities()

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

        gridworld = getattr(cls, parsed_args.id)(
            random_state=random_state
        )

        return gridworld, unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            n_rows: int,
            n_columns: int,
            terminal_states: List[Tuple[int, int]],
            RR: List[Reward]
    ):
        """
        Initialize the gridworld.

        :param name: Name.
        :param random_state: Random state.
        :param n_rows: Number of row.
        :param n_columns: Number of columns.
        :param terminal_states: List of terminal-state locations.
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

        SS = [
            ModelBasedMdpState(
                i=row_i * n_columns + col_j,
                AA=AA,
                terminal=False
            )
            for row_i in range(n_rows)
            for col_j in range(n_columns)
        ]

        for row, col in terminal_states:
            SS[row * n_columns + col].terminal = True

        super().__init__(
            name=name,
            random_state=random_state,
            SS=SS,
            RR=RR
        )

        # initialize the model within each state
        s: ModelBasedMdpState
        for s in self.SS:
            s.initialize_model(self.SS, self.RR)

        self.grid = np.array(self.SS).reshape(n_rows, n_columns)


@rl_text(chapter=4, page=84)
class GamblersProblem(MdpEnvironment):
    """
    Gambler's problem MDP environment.
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
            '--p-h',
            type=float,
            default=0.5,
            help='Probability of coin toss coming up heads.'
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

        gamblers_problem = GamblersProblem(
            name=f"gambler's problem (p={parsed_args.p_h})",
            random_state=random_state,
            **dict(parsed_args._get_kwargs())
        )

        return gamblers_problem, unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            p_h: float
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param p_h: Probability of coin toss coming up heads.
        """

        self.p_h = p_h
        self.p_t = 1 - p_h

        # the range of possible actions:  stake 0 (no play) through 50 (at capital=50). beyond a capital of 50 the
        # agent is only allowed to stake an amount that would take them to 100 on a win.
        AA = [Action(i=stake, name=f'Stake {stake}') for stake in range(0, 51)]

        # two possible rewards:  0.0 and 1.0
        self.r_not_win = Reward(0, 0.0)
        self.r_win = Reward(1, 1.0)
        RR = [self.r_not_win, self.r_win]

        # range of possible states (capital levels)
        SS = [
            ModelBasedMdpState(
                i=capital,

                # the range of permissible actions is state dependent
                AA=[
                    a
                    for a in AA
                    if a.i <= min(capital, 100 - capital)
                ],

                terminal=capital == 0 or capital == 100
            )

            # include terminal capital levels of 0 and 100
            for capital in range(0, 101)
        ]

        super().__init__(
            name=name,
            random_state=random_state,
            SS=SS,
            RR=RR
        )

        self.SS: List[ModelBasedMdpState]

        for s in self.SS:
            s.initialize_model(self.SS, self.RR)

        for s in self.SS:
            for a in s.p_S_prime_R_given_A:

                # next state and reward if heads
                s_prime_h = self.SS[s.i + a.i]
                if s_prime_h.i > 100:
                    raise ValueError('Expected state to be <= 100')

                r_h = self.r_win if not s.terminal and s_prime_h.i == 100 else self.r_not_win
                s.p_S_prime_R_given_A[a][s_prime_h][r_h] = self.p_h

                # next state and reward if tails
                s_prime_t = self.SS[s.i - a.i]
                if s_prime_t.i < 0:
                    raise ValueError('Expected state to be >= 0')

                r_t = self.r_win if not s.terminal and s_prime_t.i == 100 else self.r_not_win
                s.p_S_prime_R_given_A[a][s_prime_t][r_t] += self.p_t  # add the probability, in case the results of head and tail are the same.

        for s in self.SS:
            s.check_marginal_probabilities()
