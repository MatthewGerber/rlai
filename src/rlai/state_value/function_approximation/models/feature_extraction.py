from abc import ABC, abstractmethod
from itertools import product
from typing import List, Optional, Dict

import numpy as np

from rlai.core import MdpState
from rlai.meta import rl_text
from rlai.models.feature_extraction import FeatureExtractor, OneHotCategory, OneHotCategoricalFeatureInteracter


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


class StateDimensionSegment:
    """
    Segment of a state dimension.
    """

    @staticmethod
    def get_indicator_range() -> List[bool]:
        """
        Get list of indicators.

        :return: Indicators.
        """

        return [True, False]

    def __init__(
            self,
            dimension: int,
            low: Optional[float],
            high: Optional[float]
    ):
        """
        Initialize the segment.

        :param dimension: Dimension index.
        :param low: Low value (inclusive) of the segment.
        :param high: High value (exclusive) of the segment.
        """

        self.dimension = dimension
        self.low = low
        self.high = high

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return f'd{self.dimension}:  {"(" if self.low is None else "["}{self.low}, {self.high})'

    def get_indicator(
            self,
            state: np.ndarray
    ) -> bool:
        """
        Get indicator for a state value.

        :param state: State vector.
        :return: Indicator.
        """

        dimension_value = state[self.dimension]
        above_low = self.low is None or dimension_value >= self.low
        below_high = self.high is None or dimension_value < self.high

        return above_low and below_high


class OneHotStateSegmentFeatureInteracter:
    """
    One-hot state segment feature interacter.
    """

    def interact(
            self,
            state_matrix: np.ndarray,
            state_feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Interact a state-feature matrix with its one-hot state segment encoding.

        :param state_matrix: State matrix (#obs, #state_dimensionality).
        :param state_feature_matrix: State-feature matrix (#obs, #features).
        :return: Interacted state-feature matrix (#obs, #features * #state_segments).
        """

        # interact feature vectors per state category, where the category indicates the joint indicator of the state
        # dimension segments.
        state_categories = [
            OneHotCategory(*[
                state_dimension_segment.get_indicator(state_vector)
                for state_dimension_segment in self.state_dimension_segments
            ])
            for state_vector in state_matrix
        ]

        interacted_state_feature_matrix = self.interacter.interact(
            feature_matrix=state_feature_matrix,
            categorical_values=state_categories
        )

        return interacted_state_feature_matrix

    def __init__(
            self,
            dimension_breakpoints: Dict[int, List[float]]
    ):
        """
        Initialize the interacter.

        :param dimension_breakpoints:
        """

        self.state_dimension_segments = [
            StateDimensionSegment(dimension, low, high)
            for dimension, breakpoints in dimension_breakpoints.items()
            for low, high in zip([None] + breakpoints[:-1], breakpoints)
        ]

        self.interacter = OneHotCategoricalFeatureInteracter([
            OneHotCategory(*args)
            for args in product(*[
                state_dimension_segment.get_indicator_range()
                for state_dimension_segment in self.state_dimension_segments
            ])
        ])
