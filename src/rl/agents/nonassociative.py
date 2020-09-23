from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Dict, final, Tuple

from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.environments.state import State
from rl.meta import rl_text
from rl.utils import IncrementalSampleAverager


@rl_text(chapter=2, page=27)
class Nonassociative(Agent, ABC):
    """
    A nonassociative agent.
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
            '--initial-q-value',
            type=float,
            default=0.0,
            help='Initial Q-value to use for all actions. Use values greater than zero to encourage exploration in the early stages of the run.'
        )

        parser.add_argument(
            '--alpha',
            type=float,
            default=None,
            help='Step-size to use in incremental reward averaging. Pass None for decreasing (i.e., unweighted average) or a constant in (0, 1] for recency weighted.'
        )

        return parser.parse_known_args(unparsed_args, parsed_args)

    def reset_for_new_run(
            self
    ):
        """
        Reset the agent to a state prior to any learning.
        """

        super().reset_for_new_run()

        for averager in self.Q.values():
            averager.reset()

    @final
    def sense(
            self,
            state: State
    ):
        """
        No effect (the agent is nonassociative).

        :param state: State.
        """

        super().sense(state)

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
            AA: List[Action],
            name: str,
            random_state: RandomState,
            initial_q_value: float,
            alpha: float
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of agent.
        :param random_state: Random state.
        :param initial_q_value: Initial Q-value to use for all actions. Use values greater than zero to encourage
        exploration in the early stages of the run.
        :param alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state
        )

        self.Q: Dict[Action, IncrementalSampleAverager] = {
            a: IncrementalSampleAverager(
                initial_value=initial_q_value,
                alpha=alpha
            )
            for a in self.AA
        }
