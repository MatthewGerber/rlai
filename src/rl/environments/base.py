from abc import ABC, abstractmethod

from rl.agents.base import Agent
from rl.runners.monitor import Monitor


class Environment(ABC):

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
            name: str
    ):
        """
        Initialize the environment.
        :param name: Name of the environment.
        """

        self.name = name

    def __str__(
            self
    ):
        """
        Return name.

        :return: Name.
        """

        return self.name
