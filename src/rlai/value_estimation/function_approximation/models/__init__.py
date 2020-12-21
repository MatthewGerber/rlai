from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import Tuple, List, Any, Optional

import numpy as np
import pandas as pd

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.states.mdp import MdpState


@rl_text(chapter=9, page=197)
class FunctionApproximationModel(ABC):
    """
    Function approximation model.
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
        """
        Fit the model to a matrix of features and a vector of returns.

        :param X: Features.
        :param y: Returns.
        :param weight: Weight.
        """
        pass

    @abstractmethod
    def evaluate(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        """
        Evaluate the model at a matrix of observations.

        :param X: Observations.
        :return: Vector of evaluations.
        """
        pass

    @abstractmethod
    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the model equals another.

        :param other: Other model.
        :return: True if equal and False otherwise.
        """
        pass

    @abstractmethod
    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the model does not equal another.

        :param other: Other model.
        :return: True if not equal and False otherwise.
        """
        pass


@rl_text(chapter=9, page=197)
class FeatureExtractor(ABC):
    """
    Feature extractor.
    """

    @abstractmethod
    def extract(
            self,
            state: MdpState,
            action: Action
    ) -> pd.DataFrame:
        """
        Extract features from a state and action.

        :param state: State.
        :param action: Action.
        :return: DataFrame of features.
        """
        pass


class StateActionIdentityFeatureExtractor(FeatureExtractor):
    """
    Simple state-action identity extractor.
    """

    def extract(
            self,
            state: MdpState,
            action: Action
    ) -> pd.DataFrame:
        """
        Extract the discrete state and action identifiers.

        :param state: State.
        :param action: Action.
        :return: DataFrame.
        """

        return pd.DataFrame([
            state.i,
            action.i
        ], columns=['s', 'a'])
