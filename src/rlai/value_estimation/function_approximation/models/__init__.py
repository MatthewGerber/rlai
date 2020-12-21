from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Tuple, List, Any, Optional

import numpy as np
import pandas as pd

from rlai.actions import Action
from rlai.states.mdp import MdpState


class FunctionApproximationModel(ABC):

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
            args: List[str]
    ) -> Tuple[Any, List[str]]:
        """
        Initialize a model from arguments.

        :param args: Arguments.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """
        pass

    @abstractmethod
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            weight: Optional[float]
    ):
        pass

    @abstractmethod
    def evaluate(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
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


class FeatureExtractor(ABC):

    @abstractmethod
    def extract(
            self,
            state: MdpState,
            action: Action
    ) -> pd.DataFrame:
        pass


class StateActionIdentityFeatureExtractor(FeatureExtractor):

    def extract(
            self,
            state: MdpState,
            action: Action
    ) -> pd.DataFrame:

        return pd.DataFrame([
            state.i,
            action.i
        ], columns=['s', 'a'])
