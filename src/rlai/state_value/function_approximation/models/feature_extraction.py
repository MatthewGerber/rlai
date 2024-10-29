from abc import ABC, abstractmethod
from itertools import product
from typing import List, Optional, Dict, Callable, Any

import numpy as np

from rlai.core import MdpState
from rlai.docs import rl_text
from rlai.models.feature_extraction import FeatureExtractor, OneHotCategory, OneHotCategoricalFeatureInteracter


@rl_text(chapter='Feature Extractors', page=1)
class StateFeatureExtractor(FeatureExtractor, ABC):
    """
    Feature extractor for states.
    """

    def __init__(
            self,
            scale_features: bool
    ):
        """
        Initialize the extractor.

        :param scale_features: Whether to scale features.
        """

        self.scale_features = scale_features

    @abstractmethod
    def extract(
            self,
            states: List[MdpState],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param states: States.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature matrix (#states, #features).
        """


class StateIndicator(ABC):
    """
    Abstract state indicator for one-hot feature encoding.
    """

    @abstractmethod
    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

    @abstractmethod
    def get_range(
            self
    ) -> List[Any]:
        """
        Get the range (possible values) of the current indicator.

        :return: Range of values.
        """

    @abstractmethod
    def get_value(
            self,
            state_vector: np.ndarray
    ) -> Any:
        """
        Get the value of the current indicator for a state.

        :param state_vector: State vector.
        :return: Value, which must be in the range returned by `get_range`.
        """


class StateLambdaIndicator(StateIndicator):
    """
    State indicator via lambda function.
    """

    def __init__(
            self,
            function: Callable[[np.ndarray], Any],
            function_range: List[Any]
    ):
        """
        Initialize the indicator.

        :param function: Function to apply to states.
        :param function_range: Range of function.
        """

        self.function = function
        self.function_range = function_range

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return '<function>'

    def get_range(
            self
    ) -> List[Any]:
        """
        Get the range (possible values) of the current indicator.

        :return: Range of values.
        """

        return self.function_range

    def get_value(
            self,
            state_vector: np.ndarray
    ) -> Any:
        """
        Get the value of the current indicator for a state.

        :param state_vector: State vector.
        :return: Value, which must be in the range returned by `get_range`.
        """

        return self.function(state_vector)


class StateDimensionIndicator(StateIndicator, ABC):
    """
    State-dimension indicator.
    """

    def __init__(
            self,
            dimension: int
    ):
        """
        Initialize the indicator.

        :param dimension: Dimension.
        """

        self.dimension = dimension


class StateDimensionSegment(StateDimensionIndicator):
    """
    Indicates a segment of a state dimension.
    """

    @staticmethod
    def get_segments(
            dimension_breakpoints: Dict[int, List[float]]
    ) -> List[StateIndicator]:
        """
        Get segments for a dictionary of breakpoints

        :param dimension_breakpoints: Breakpoints keyed on dimensions with breakpoints as values.
        """

        return [
            StateDimensionSegment(dimension, low, high)
            for dimension, breakpoints in dimension_breakpoints.items()
            for low, high in zip([None] + breakpoints[:-1], breakpoints)  # type: ignore[operator]
        ]

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

        super().__init__(dimension)

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

    def get_range(
            self
    ) -> List[Any]:
        """
        Get the range (possible values) of the current indicator.

        :return: Range of values.
        """

        return [True, False]

    def get_value(
            self,
            state_vector: np.ndarray
    ) -> Any:
        """
        Get the value of the current indicator for a state.

        :param state_vector: State vector.
        :return: Value.
        """

        dimension_value = float(state_vector[self.dimension])

        above_low = self.low is None or dimension_value >= self.low
        below_high = self.high is None or dimension_value < self.high

        return above_low and below_high


class StateDimensionLambda(StateDimensionIndicator):
    """
    Lambda applied to a state dimension.
    """

    def __init__(
            self,
            dimension: int,
            function: Callable[[float], Any],
            function_range: List[Any]
    ):
        """
        Initialize the segment.

        :param dimension: Dimension.
        :param function: Function to apply to values in the given dimension.
        :param function_range: Range of function.
        """

        super().__init__(dimension)

        self.function = function
        self.function_range = function_range

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return f'd{self.dimension}:  <function>'

    def get_range(self) -> List[Any]:
        """
        Get the range (possible values) of the current indicator.

        :return: Range of values.
        """

        return self.function_range

    def get_value(
            self,
            state_vector: np.ndarray
    ) -> Any:
        """
        Get the value of the current indicator for a state.

        :param state_vector: State vector.
        :return: Value.
        """

        return self.function(float(state_vector[self.dimension]))


class OneHotStateIndicatorFeatureInteracter:
    """
    One-hot state indicator feature interacter.
    """

    def interact(
            self,
            state_matrix: np.ndarray,
            state_feature_matrix: np.ndarray,
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Interact a state-feature matrix with its one-hot state-indicator encoding.

        :param state_matrix: State matrix (#obs, #state_dimensionality), from which to derive indicators.
        :param state_feature_matrix: State-feature matrix (#obs, #features).
        :param refit_scaler: Whether to refit the scaler.
        :return: Interacted state-feature matrix (#obs, #features * #joint_indicators).
        """

        # interact feature vectors per state category, where the category indicates the joint indicator of the state.
        state_categories = [
            OneHotCategory(*[
                indicator.get_value(state_vector)
                for indicator in self.indicators
            ])
            for state_vector in state_matrix
        ]

        # use optional scaling
        interacted_state_feature_matrix = self.interacter.interact(
            feature_matrix=state_feature_matrix,
            categorical_values=state_categories,
            refit_scaler=refit_scaler
        )

        return interacted_state_feature_matrix

    def __init__(
            self,
            indicators: List[StateIndicator],
            scale_features: bool
    ):
        """
        Initialize the interacter.

        :param indicators: State-dimension indicators.
        :param scale_features: Whether to scale features.
        """

        self.indicators = indicators

        self.interacter = OneHotCategoricalFeatureInteracter(
            categories=[
                OneHotCategory(*args)
                for args in product(*[
                    indicator.get_range()
                    for indicator in self.indicators
                ])
            ],
            scale_features=scale_features
        )
