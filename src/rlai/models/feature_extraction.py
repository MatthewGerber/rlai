from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import List, Tuple, Any, Optional

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.utils import get_base_argument_parser


@rl_text(chapter=9, page=197)
class FeatureExtractor(ABC):
    """
    Feature extractor.
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
    ) -> Tuple[Any, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

    def get_feature_names(
            self
    ) -> List[str]:
        """
        Get names of features.

        :return: List of feature names.
        """

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """


@rl_text(chapter='Feature Extractors', page=1)
class NonstationaryFeatureScaler:
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

    def scale_features(
            self,
            X: np.ndarray,
            for_fitting: bool
    ) -> np.ndarray:
        """
        Scale features.

        :param X: Feature matrix.
        :param for_fitting: Whether the extracted features will be used for fitting (True) or prediction (False).
        :return: Scaled feature matrix.
        """

        # only fit the scaler if the features will be for fitting the model. if they will be for prediction, then we
        # should use whatever scaling parameters were obtained for fitting, as that's what the model coefficients are
        # calibrated for.
        if for_fitting:

            # append feature matrix to the refit history
            if self.refit_history is None:
                self.refit_history = X
            else:
                self.refit_history = np.append(self.refit_history, X, axis=0)

            # refit scaler if we've extracted enough
            self.num_observations += X.shape[0]
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
                num_rows_to_delete = max(history_length + self.num_observations_refit_feature_scaler - self.refit_history_length, 0)
                num_rows_to_delete = min(num_rows_to_delete, self.refit_history.shape[0])
                if num_rows_to_delete > 0:
                    rows_to_delete = list(range(num_rows_to_delete))
                    self.refit_history = np.delete(self.refit_history, rows_to_delete, axis=0)

                self.num_observations = 0

            # otherwise, just run a partial fitting without sample weight. in principle, the weight would be
            # exponentially increasing following the first refit, but we'll quickly overflow. this is a compromise, and
            # it seems to work. it also addresses the case where the scaler has not yet been refit.
            else:
                self.feature_scaler.partial_fit(X)

        # scale features
        try:
            X = self.feature_scaler.transform(X)

        # the following exception will be thrown if the scaler has not yet been fitted. catch and ignore scaling.
        except NotFittedError:
            pass

        return X

    def __init__(
            self,
            num_observations_refit_feature_scaler: int,
            refit_history_length: int,
            refit_weight_decay: float
    ):
        """
        Initializer the scaler.

        :param num_observations_refit_feature_scaler: Number of observations to collect before refitting.
        :param refit_history_length: Number of observations to use in the refitting.
        :param refit_weight_decay: Exponential weight decay for the observations used in the refitting.
        """

        self.num_observations_refit_feature_scaler = num_observations_refit_feature_scaler
        self.refit_history_length = refit_history_length
        self.refit_weight_decay = refit_weight_decay

        self.refit_history: Optional[np.ndarray] = None
        self.feature_scaler = StandardScaler()
        self.num_observations = 0
