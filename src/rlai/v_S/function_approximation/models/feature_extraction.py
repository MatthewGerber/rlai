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
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :param refit_scaler: Whether or not to refit the feature scaler before scaling the extracted features.
        :return: State-feature vector.
        """
