from abc import ABC, abstractmethod
from typing import List, Set

from rl.agents.action import Action
from rl.environments.state import State


class Agent(ABC):

    @abstractmethod
    def reset(
            self
    ):
        """
        Reset the agent to a state prior to any learning.
        """
        pass

    @abstractmethod
    def sense(
            self,
            state: State
    ):
        """
        Pass the agent state information to sense.

        :param state: State.
        """
        pass

    @abstractmethod
    def act(
            self
    ) -> Action:
        """
        Request an action from the agent.
        :return: Action
        """
        pass

    @abstractmethod
    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward.
        """
        pass

    def __init__(
            self,
            AA: List[Action]
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        """

        self.AA = AA
