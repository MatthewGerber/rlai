from abc import ABC, abstractmethod
from typing import List, Optional, final

from numpy.random import RandomState

from rl.agents.action import Action
from rl.environments.state import State


class Agent(ABC):

    def reset_for_new_run(
            self
    ):
        """
        Reset the agent to a state prior to any learning.
        """

        self.most_recent_action = None

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

    @final
    def act(
            self
    ) -> Action:
        """
        Request an action from the agent.

        :return: Action
        """

        a = self.__act__()
        self.most_recent_action = a
        return a

    @abstractmethod
    def __act__(
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
            AA: List[Action],
            name: str,
            random_state: RandomState
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of the agent.
        :param random_state: Random State.
        """

        self.AA = AA
        self.name = name
        self.random_state = random_state

        self.most_recent_action: Optional[Action] = None

    def __str__(
            self
    ):
        """
        Return name.

        :return: Name.
        """

        return self.name
