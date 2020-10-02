from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Tuple, List, final, Optional

import numpy as np
from numpy.random import RandomState

from rl.actions import Action
from rl.states import State


class Agent(ABC):
    """
    Base class for all agents.
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
            environment,  # can't provide Environment type hint due to circular reference with Agent
            random_state: RandomState
    ) -> List:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param environment: Environment.
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
        self.most_recent_action_tick = None
        self.N_t_A = np.zeros_like(self.N_t_A)

    def sense(
            self,
            state: State,
            t: int
    ):
        """
        Pass the agent state information to sense.

        :param state: State.
        :param t: Time tick for `state`.
        """

        self.most_recent_state = state
        self.most_recent_state_tick = t

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

        if a is None:
            raise ValueError('Agent returned action of None.')

        self.most_recent_action = a
        self.most_recent_action_tick = t
        self.N_t_A[a.i] += 1

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

        :param AA: List of all possible actions, with identifiers sorted in increasing order from zero.
        :param name: Name of the agent.
        :param random_state: Random state.
        """

        for i, a in enumerate(AA):
            if a.i != i:
                raise ValueError('Actions must be sorted in increasing order from zero.')

        self.AA = AA
        self.name = name
        self.random_state = random_state

        self.most_recent_action: Optional[Action] = None
        self.most_recent_action_tick: Optional[int] = None
        self.most_recent_state: Optional[State] = None
        self.most_recent_state_tick: Optional[int] = None
        self.N_t_A: np.ndarray = np.zeros(len(self.AA))

    def __str__(
            self
    ):
        """
        Return name.

        :return: Name.
        """

        return self.name
