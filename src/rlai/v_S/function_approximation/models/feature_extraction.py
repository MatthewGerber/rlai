from abc import ABC, abstractmethod

import numpy as np

from rlai.meta import rl_text
from rlai.models.feature_extraction import FeatureExtractor
from rlai.states.mdp import MdpState


@rl_text(chapter='Feature Extractors', page=1)
class StateFeatureExtractor(FeatureExtractor, ABC):
    """
    Feature extractor for states.
    """

    @abstractmethod
    def get_state_space_dimensionality(
            self
    ) -> int:
        """
        Get the state-space dimensionality.

        :return: Dimensions.
        """

    @abstractmethod
    def get_action_space_dimensionality(
            self
    ) -> int:
        """
        Get the action-space dimensionality.

        :return: Dimensions.
        """

    @abstractmethod
    def extract(
            self,
            state: MdpState,
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :return: State-feature vector.
        """
