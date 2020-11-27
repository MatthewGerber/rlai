from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Tuple, List, Any, final, Optional

from numpy.random import RandomState

from rlai.agents import Agent
from rlai.runners.monitor import Monitor
from rlai.states import State


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

        parser.add_argument(
            '--T',
            type=int,
            help='Maximum number of time steps to run.'
        )

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
    def reset_for_new_run(
            self,
            agent: Agent
    ) -> Optional[State]:
        """
        Reset the the environment.

        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: New state.
        """

        self.num_resets += 1

        return None

    @final
    def run(
            self,
            agent,  # can't provide Agent type hint due to circular reference with Environment
            monitor: Monitor
    ):
        """
        Run the environment with an agent.

        :param agent: Agent to run.
        :param monitor: Monitor.
        """

        t = 0
        have_T = self.T is not None
        while True:

            terminate = self.run_step(t, agent, monitor)

            t += 1

            if terminate or (have_T and t >= self.T):
                break

    @abstractmethod
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
        pass

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int]
    ):
        """
        Initialize the environment.

        :param name: Name of the environment.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        """

        self.name = name
        self.random_state = random_state
        self.T = T

        self.num_resets = 0

    def __str__(
            self
    ):
        """
        Return name.

        :return: Name.
        """

        return self.name
