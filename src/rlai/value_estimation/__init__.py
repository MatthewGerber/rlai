from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Optional, Iterable, Tuple, List, Any, Iterator

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.policies import Policy
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

    @abstractmethod
    def __len__(
            self
    ) -> int:
        pass

    @abstractmethod
    def __iter__(
            self
    ) -> Iterator[Action]:
        pass

    @abstractmethod
    def __contains__(
            self,
            action: Action
    ) -> bool:
        pass


class StateActionValueEstimator(ABC):

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

        # future arguments for this base class can be added here...

        return parser.parse_known_args(args)

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[Any, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """
        pass

    @abstractmethod
    def get_initial_policy(
            self
    ) -> Policy:
        pass

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

    @abstractmethod
    def __len__(
            self
    ) -> int:
        pass

    @abstractmethod
    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        pass

    @abstractmethod
    def __eq__(
            self,
            other
    ) -> bool:
        pass

    @abstractmethod
    def __ne__(
            self,
            other
    ) -> bool:
        pass
