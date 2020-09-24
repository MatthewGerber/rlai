from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import final, Tuple, List

from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.environments.state import State


class Nonassociative(Agent, ABC):
    """
    Abstract base class for all nonassociative agents.
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
            '--alpha',
            type=float,
            default=None,
            help='Step-size.'
        )

        return parser.parse_known_args(unparsed_args, parsed_args)

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

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            alpha: float
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of agent.
        :param random_state: Random state.
        :param alpha: Step-size parameter.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state
        )

        self.alpha = alpha
