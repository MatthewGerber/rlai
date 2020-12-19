from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from rlai.actions import Action
from rlai.states.mdp import MdpState


class FunctionApproximationModel(ABC):

    @abstractmethod
    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
            weight: float
    ):
        pass

    @abstractmethod
    def evaluate(
            self,
            X: np.ndarray,
    ) -> np.ndarray:
        pass


class FeatureExtractor(ABC):

    @abstractmethod
    def extract(
            self,
            state: MdpState,
            action: Action
    ) -> pd.DataFrame:
        pass
