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


class ActionValueEstimator(ABC):

    @abstractmethod
    def __contains__(
            self,
            action: Action
    ) -> bool:
        pass

    @abstractmethod
    def __getitem__(
            self,
            action: Action
    ) -> ValueEstimator:
        pass

    @abstractmethod
    def __setitem__(
            self,
            action: Action,
            value_estimator: ValueEstimator
    ):
        pass

    @abstractmethod
    def __len__(
            self
    ) -> int:
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
    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        pass

    @abstractmethod
    def __getitem__(
            self,
            state: MdpState
    ) -> ActionValueEstimator:
        pass

    @abstractmethod
    def __setitem__(
            self,
            state: MdpState,
            action_value_estimator: ActionValueEstimator
    ):
        pass

    @abstractmethod
    def __len__(
            self
    ) -> int:
        pass
