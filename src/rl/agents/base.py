from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from typing import List, Optional, final, Dict, Tuple

from numpy.random import RandomState

from rl.agents.action import Action
from rl.environments.state import State


class Agent(ABC):
    """
    Abstract base class for all agents.
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

        parser = ArgumentParser(allow_abbrev=False)

        return parser.parse_known_args(args)

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            AA: List[Action],
            random_state: RandomState
    ) -> List:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param AA: List of possible actions.
        :param random_state: Random state.
        :return: List of agents.
        """
        pass

    def reset_for_new_run(
            self
    ):
        """
        Reset the agent to a state prior to any learning.
        """

        self.most_recent_action = None
        self.N_t_A.update({
            a: 0
            for a in self.N_t_A
        })

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
            self,
            t
    ) -> Action:
        """
        Request an action from the agent.

        :param t: Current time step.
        :return: Action
        """

        a = self.__act__(t=t)

        self.most_recent_action = a
        self.N_t_A[a] += 1

        return a

    @abstractmethod
    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Request an action from the agent.

        :param t: Current time step.
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
        :param random_state: Random state.
        """

        self.AA = AA
        self.name = name
        self.random_state = random_state

        self.most_recent_action: Optional[Action] = None
        self.N_t_A: Dict[Action, int] = {
            a: 0
            for a in self.AA
        }

    def __str__(
            self
    ):
        """
        Return name.

        :return: Name.
        """

        return self.name
