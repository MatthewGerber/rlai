from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Tuple, List, Any, final

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
            self
    ) -> State:
        """
        Reset the the environment.

        :return: New state.
        """
        pass

    @final
    def run(
            self,
            agent,  # can't provide Agent type hint due to circular reference with Environment
            T: int,
            monitor: Monitor
    ):
        """
        Run the environment with an agent.

        :param agent: Agent to run.
        :param T: Number of time steps to run.
        :param monitor: Monitor.
        """

        # use any to short-circuit the runs if any step returns True (for termination)
        any(
            self.run_step(t, agent, monitor)
            for t in range(T)
        )

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
            random_state: RandomState
    ):
        """
        Initialize the environment.

        :param name: Name of the environment.
        :param random_state: Random state.
        """

        self.name = name
        self.random_state = random_state

    def __str__(
            self
    ):
        """
        Return name.

        :return: Name.
        """

        return self.name
