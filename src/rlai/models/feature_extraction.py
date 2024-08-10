from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import List, Tuple, Any, Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from rlai.core import MdpState
from rlai.core.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.utils import get_base_argument_parser


@rl_text(chapter=9, page=197)
class FeatureExtractor(ABC):
    """
    Base feature extractor for all others. This class does not define any extraction function, since the signature of
    such a function depends on what, conceptually, we're extracting features from. The definition of this signature is
    deferred to inheriting classes that are closer to their conceptual extraction targets.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        return get_base_argument_parser()

    @classmethod
    @abstractmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple['FeatureExtractor', List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

    def get_feature_names(
            self
    ) -> Optional[List[str]]:
        """
        Get names of features.

        :return: List of feature names.
        """

        return None

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset the feature extractor for a new run.

        :param state: Initial state.
        """

    @abstractmethod
    def extracts_intercept(
            self
    ) -> bool:
        """
        Whether the feature extractor extracts an intercept (constant) term.

        :return: True if an intercept (constant) term is extracted and False otherwise.
        """

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """


@rl_text(chapter='Feature Extractors', page=1)
class FeatureScaler(ABC):
    """
    Base class for all feature scalers.
    """

    @abstractmethod
    def scale_features(
            self,
            feature_matrix: np.ndarray,
            refit_before_scaling: bool
    ) -> np.ndarray:
        """
        Scale features.

        :param feature_matrix: Feature matrix.
        :param refit_before_scaling: Whether to refit the scaler using `feature_matrix` before scaling.
        :return: Scaled feature matrix.
        """

    @abstractmethod
    def invert_scaled_features(
            self,
            feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Invert scaled features back to their original representation.

        :param feature_matrix: Feature matrix.
        :return: Feature matrix in original representation.
        """


@rl_text(chapter='Feature Extractors', page=1)
class StationaryFeatureScaler(FeatureScaler):
    """
    Stationary feature scaler.
    """

    def __init__(
            self
    ):
        """
        Initialize the scaler.
        """

        self.feature_scaler = StandardScaler()

    def scale_features(
            self,
            feature_matrix: np.ndarray,
            refit_before_scaling: bool
    ) -> np.ndarray:
        """
        Scale features.

        :param feature_matrix: Feature matrix.
        :param refit_before_scaling: Whether to refit the scaler using `feature_matrix` before scaling.
        :return: Scaled feature matrix.
        """

        if refit_before_scaling:
            self.feature_scaler.partial_fit(feature_matrix)

        try:

            scaled_feature_matrix = self.feature_scaler.transform(feature_matrix)

        # the following exception will be thrown if the scaler has not yet been fitted. catch and ignore scaling.
        except NotFittedError:
            scaled_feature_matrix = feature_matrix

        return scaled_feature_matrix

    def invert_scaled_features(
            self,
            feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Invert scaled features back to their original representation.

        :param feature_matrix: Feature matrix.
        :return: Feature matrix in original representation.
        """

        try:

            inverted_feature_matrix = self.feature_scaler.inverse_transform(feature_matrix)

        # the following exception will be thrown if the scaler has not yet been fitted. catch and ignore scaling.
        except NotFittedError:
            inverted_feature_matrix = feature_matrix

        return inverted_feature_matrix


@rl_text(chapter='Feature Extractors', page=1)
class NonstationaryFeatureScaler(FeatureScaler):
    """
    It is common for function approximation models to require some sort of state-feature scaling in order to converge
    upon optimal solutions. For example, in stochastic gradient descent, the use of state features with different scales
    can cause weight updates to increase loss depending on the step size and gradients of the loss function with respect
    to the weights. A common approach to scaling weights is standardization, and scikit-learn supports this with the
    StandardScaler. However, the StandardScaler is intended for use with stationary state-feature distributions, whereas
    the state-feature distributions in RL tasks can be nonstationary (e.g., a cartpole agent that moves through distinct
    state-feature distributions over the course of learning). This class provides a simple extension of the
    StandardScaler to address nonstationary state-feature scaling. It refits the scaler periodically using the most
    recent state-feature observations. Furthermore, it assigns weights to these observations that decay exponentially
    with the observation age.
    """

    def __init__(
            self,
            num_observations_refit_feature_scaler: int,
            refit_history_length: int,
            refit_weight_decay: float
    ):
        """
        Initialize the scaler.

        :param num_observations_refit_feature_scaler: Number of observations to collect before refitting.
        :param refit_history_length: Number of observations to use when refitting the feature scaler.
        :param refit_weight_decay: Exponential weight decay for the observations used in refitting the feature scaler.
        """

        self.num_observations_refit_feature_scaler = num_observations_refit_feature_scaler
        self.refit_history_length = refit_history_length
        self.refit_weight_decay = refit_weight_decay

        self.refit_history: Optional[np.ndarray] = None
        self.feature_scaler = StandardScaler()
        self.num_observations = 0

    def scale_features(
            self,
            feature_matrix: np.ndarray,
            refit_before_scaling: bool
    ) -> np.ndarray:
        """
        Scale features.

        :param feature_matrix: Feature matrix.
        :param refit_before_scaling: Whether to refit the scaler using `feature_matrix` before scaling.
        :return: Scaled feature matrix.
        """

        if refit_before_scaling:

            # append feature matrix to the refit history
            if self.refit_history is None:
                self.refit_history = feature_matrix
            else:
                self.refit_history = np.append(self.refit_history, feature_matrix, axis=0)

            # refit scaler if we've extracted enough
            self.num_observations += feature_matrix.shape[0]
            if self.num_observations >= self.num_observations_refit_feature_scaler:

                # get recent history up to specified length (note that the rows will be ordered most recent first)
                history_length = self.refit_history.shape[0]
                num_history_rows = min(history_length, self.refit_history_length)
                history_rows = list(reversed(range(history_length)))[:num_history_rows]
                history_to_fit = self.refit_history[history_rows]

                # get sample weights, with most recent receiving a weight of 1 and decaying exponentially
                sample_weights = self.refit_weight_decay ** np.array(list(range(num_history_rows)))

                # fit new scaler
                self.feature_scaler = StandardScaler()
                self.feature_scaler.fit(history_to_fit, sample_weight=sample_weights)

                # delete old rows from history and reset number of extractions
                num_rows_to_delete = max(
                    history_length + self.num_observations_refit_feature_scaler - self.refit_history_length,
                    0
                )
                num_rows_to_delete = min(num_rows_to_delete, self.refit_history.shape[0])
                if num_rows_to_delete > 0:
                    rows_to_delete = list(range(num_rows_to_delete))
                    self.refit_history = np.delete(self.refit_history, rows_to_delete, axis=0)

                self.num_observations = 0

            # otherwise, just run a partial fitting without sample weight. in principle, the weight would be
            # exponentially increasing following the first refit, but we'll quickly overflow. this is a compromise, and
            # it seems to work. it also addresses the case where the scaler has not yet been refit.
            else:
                self.feature_scaler.partial_fit(feature_matrix)

        try:

            scaled_feature_matrix = self.feature_scaler.transform(feature_matrix)

        # the following exception will be thrown if the scaler has not yet been fitted. catch and ignore scaling.
        except NotFittedError:
            scaled_feature_matrix = feature_matrix

        return scaled_feature_matrix

    def invert_scaled_features(
            self,
            feature_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Invert scaled features back to their original representation.

        :param feature_matrix: Feature matrix.
        :return: Feature matrix in original representation.
        """

        try:

            inverted_feature_matrix = self.feature_scaler.inverse_transform(feature_matrix)

        # the following exception will be thrown if the scaler has not yet been fitted. catch and ignore scaling.
        except NotFittedError:
            inverted_feature_matrix = feature_matrix

        return inverted_feature_matrix


@rl_text(chapter='Feature Extractors', page=1)
class OneHotCategoricalFeatureInteracter:
    """
    Feature interacter for one-hot encoded categorical values.
    """

    def interact(
            self,
            feature_matrix: np.ndarray,
            categorical_values: List[Any]
    ) -> np.ndarray:
        """
        Perform one-hot interaction of a matrix of feature vectors with associated categorical levels.

        :param feature_matrix: Feature matrix (#obs, #features).
        :param categorical_values: List of categorical levels, with length equal to #obs.
        :return: Interacted feature matrix (#obs, #features * #levels).
        """

        num_rows = feature_matrix.shape[0]
        num_cats = len(categorical_values)
        if num_rows != num_cats:
            raise ValueError(f'Expected {num_rows} categorical values but got {num_cats}')

        num_features = feature_matrix.shape[1]
        interacted_state_features = np.zeros((num_rows, num_features * len(self.category_idx)))
        for i, feature_vector in enumerate(feature_matrix):
            cat_idx = self.category_idx[categorical_values[i]]
            start_idx = cat_idx * num_features
            end_idx = start_idx + num_features - 1
            interacted_state_features[i, start_idx:end_idx + 1] = feature_vector

        return interacted_state_features

    def __init__(
            self,
            categories: List[Any]
    ):
        """
        Initialize the interacter.

        :param categories: List of categories that will be one-hot encoded. These can be of any type that is hashable.
        See `rlai.models.feature_extraction.OneHotCategory` for a general-purpose category class.
        """

        self.category_idx = {
            category: i
            for i, category in enumerate(categories)
        }


@rl_text(chapter='Feature Extractors', page=1)
class OneHotCategory:
    """
    General-purpose category specification. Instances of this class are passed to
    `rlai.models.feature_extraction.OneHotCategoricalFeatureInteracter` to achieve one-hot encoding of feature vectors.
    """

    def __init__(
            self,
            *args
    ):
        """
        Initialize the category.

        :param args: Arguments that comprise the category. Each argument will be cast to a string, and the resulting
        strings will be concatenated to form the category identifier.
        """

        self.id = '_'.join(str(arg) for arg in args)

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check equality.

        :param other: Other category.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, OneHotCategory):
            raise ValueError(f'Expected {OneHotCategory}')

        return self.id == other.id

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check inequality.

        :param other: Other category.
        :return: True if unequal and False otherwise.
        """

        return not (self == other)

    def __hash__(
            self
    ) -> int:
        """
        Get hash code.

        :return: Hash code.
        """

        return hash(self.id)

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return self.id
