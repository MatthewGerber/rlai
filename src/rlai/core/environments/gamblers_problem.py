from argparse import ArgumentParser
from typing import List, Tuple, Optional

from numpy.random import RandomState

from rlai.core import Reward, Action, MdpState, Environment
from rlai.core.environments.mdp import ModelBasedMdpEnvironment
from rlai.docs import rl_text
from rlai.utils import parse_arguments


@rl_text(chapter=4, page=84)
class GamblersProblem(ModelBasedMdpEnvironment):
    """
    Gambler's problem MDP environment.
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
            '--p-h',
            type=float,
            default=0.5,
            help='Probability of coin toss coming up heads.'
        )

        return parser

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

        parsed_args, unparsed_args = parse_arguments(cls, args)

        gamblers_problem = cls(
            name=f"gambler's problem (p={parsed_args.p_h})",
            random_state=random_state,
            **vars(parsed_args)
        )

        return gamblers_problem, unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            p_h: float
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
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
            MdpState(
                i=capital,

                # the range of permissible actions is state dependent
                AA=[
                    a
                    for a in AA
                    if a.i <= min(capital, 100 - capital)
                ],

                terminal=capital == 0 or capital == 100,
                truncated=False
            )

            # include terminal capital levels of 0 and 100
            for capital in range(0, 101)
        ]

        super().__init__(
            name=name,
            random_state=random_state,
            T=T,
            SS=SS,
            RR=RR
        )

        for s in self.SS:
            for a in self.p_S_prime_R_given_S_A[s]:

                # next state and reward if heads
                assert s.i is not None
                s_prime_h = self.SS[s.i + a.i]
                r_h = self.r_win if not s.terminal and s_prime_h.i == 100 else self.r_not_win
                self.p_S_prime_R_given_S_A[s][a][s_prime_h][r_h] = self.p_h

                # next state and reward if tails
                assert s.i is not None
                s_prime_t = self.SS[s.i - a.i]
                r_t = self.r_win if not s.terminal and s_prime_t.i == 100 else self.r_not_win
                self.p_S_prime_R_given_S_A[s][a][s_prime_t][r_t] += self.p_t  # add the probability, in case the results of head and tail are the same.

        self.check_marginal_probabilities()
