from abc import ABC, abstractmethod

import numpy as np

from rlai.core import MdpState
from rlai.meta import rl_text
from rlai.models.feature_extraction import FeatureExtractor


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
        :param refit_scaler: Whether or not to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature vector.
        """