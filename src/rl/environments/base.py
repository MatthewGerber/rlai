from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import List, Tuple, Any

from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.runners.monitor import Monitor


class Environment(ABC):

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

        parser = ArgumentParser(allow_abbrev=False)

        return parser.parse_known_args(args)

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Any, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """
        pass

    @abstractmethod
    def run(
            self,
            agent: Agent,
            T: int,
            monitor: Monitor
    ):
        """
        Run the environment with an agent.

        :param agent: Agent to run.
        :param T: Number of time steps to run.
        :param monitor: Monitor.
        """
        pass

    def __init__(
            self,
            name: str,
            AA: List[Action],
            random_state: RandomState
    ):
        """
        Initialize the environment.
        :param name: Name of the environment.
        :param AA: List of all possible actions.
        :param random_state: Random state.
        """

        self.name = name
        self.AA = AA
        self.random_state = random_state

    def __str__(
            self
    ):
        """
        Return name.

        :return: Name.
        """

        return self.name
