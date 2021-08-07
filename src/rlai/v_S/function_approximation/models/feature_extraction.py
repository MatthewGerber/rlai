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
    def extract(
            self,
            state: MdpState,
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :return: State-feature vector.
        """
