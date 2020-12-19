from abc import ABC, abstractmethod
from typing import Optional, Iterable

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.states.mdp import MdpState


class ValueEstimator(ABC):

    @abstractmethod
    def update(
            self,
            value: float,
            weight: Optional[float] = None
    ):
        pass

    @abstractmethod
    def get_value(
            self
    ) -> float:
        pass

    def __str__(
            self
    ) -> str:

        return str(self.get_value())


class ActionValueEstimator(ABC):

    @abstractmethod
    def __getitem__(
            self,
            action: Action
    ) -> ValueEstimator:
        pass


class StateActionValueEstimator(ABC):

    @abstractmethod
    def initialize(
            self,
            state: MdpState,
            a: Action,
            alpha: Optional[float],
            weighted: bool
    ):
        pass

    @abstractmethod
    def update_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            epsilon: float
    ) -> int:
        pass

    @abstractmethod
    def __getitem__(
            self,
            state: MdpState
    ) -> ActionValueEstimator:
        pass
