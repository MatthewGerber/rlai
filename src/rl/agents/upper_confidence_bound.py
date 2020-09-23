import math
import sys
from argparse import Namespace, ArgumentParser
from typing import List, Tuple

from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.agents.nonassociative import Nonassociative
from rl.meta import rl_text


@rl_text(chapter=2, page=35)
class UpperConfidenceBound(Nonassociative):

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
            '--c',
            type=float,
            nargs='+',
            help='Space-separated list of confidence levels (higher gives more exploration).'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            AA: List[Action],
            random_state: RandomState
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param AA: List of possible actions.
        :param random_state: Random state.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        # grab and delete c values from parsed arguments
        c_values = parsed_args.c
        del parsed_args.c

        # initialize agents
        agents = [
            UpperConfidenceBound(
                AA=AA,
                name=f'UCB (c={c})',
                random_state=random_state,
                c=c,
                **dict(parsed_args._get_kwargs())
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

        n_t_a = self.N_t_a[a]

        if n_t_a == 0:
            return sys.float_info.min
        else:
            return n_t_a

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

        return max(self.AA, key=lambda a: self.Q[a].get_value() + self.c * math.sqrt(math.log(t + 1) / self.get_denominator(a)))

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            initial_q_value: float,
            alpha: float,
            c: float
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of agent.
        :param random_state: Random state.
        :param initial_q_value: Initial Q-value to use for all actions. Use values greater than zero to encourage
        exploration in the early stages of the run.
        :param alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        :param c: Confidence.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state,
            initial_q_value=initial_q_value,
            alpha=alpha
        )

        self.c = c
