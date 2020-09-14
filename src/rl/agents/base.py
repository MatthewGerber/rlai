from abc import ABC, abstractmethod
from typing import List

from rl.environments.state import State


class Agent(ABC):

    @abstractmethod
    def reset(
            self
    ):
        pass

    @abstractmethod
    def sense(
            self,
            state: State
    ):
        pass

    @abstractmethod
    def act(
            self
    ) -> int:
        pass

    @abstractmethod
    def reward(
            self,
            r: float
    ):
        pass

    def __init__(
            self,
            AA: List[int]
    ):
        self.AA = AA
