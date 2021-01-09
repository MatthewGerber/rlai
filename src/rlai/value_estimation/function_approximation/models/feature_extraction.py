from abc import ABC, abstractmethod
from argparse import ArgumentParser
from itertools import product
from typing import List, Tuple, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.utils import get_base_argument_parser, parse_arguments


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
        pass

    @abstractmethod
    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            for_fitting: bool
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param for_fitting: Whether the extracted features will be used for fitting (True) or prediction (False).
        :return: State-feature pandas.DataFrame or numpy.ndarray.
        """
        pass

    @staticmethod
    def check_state_and_action_lists(
            states: List[MdpState],
            actions: List[Action]
    ):
        """
        Check lengths of the state and action lists. Will raise exception if list lengths are not equal.

        :param states: States.
        :param actions: Actions.
        """

        num_states = len(states)
        num_actions = len(actions)
        if num_states != num_actions:
            raise ValueError(f'Expected {num_states} actions but got {num_actions}')

    @abstractmethod
    def get_feature_names(
            self
    ) -> List[str]:
        """
        Get names of extracted features.

        :return: List of feature names.
        """
        pass

    def __init__(
            self,
            environment: MdpEnvironment
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        self.environment = environment


@rl_text(chapter='Feature Extractors', page=1)
class StateActionInteractionFeatureExtractor(FeatureExtractor, ABC):
    """
    A feature extractor that extracts features comprising the interaction (in a statistical modeling sense) of
    state features with categorical actions. Categorical actions are coded as one-hot vectors with length equal to the
    number of possible discrete actions. To arrive at the full vector expression for a particular state-action pair, we
    first form the cartesian product of (a) the one-hot action vector and (b) the state features. Each pair in this
    product is then multiplied to arrive at the full vector expression of the state-action pair.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        return parser

    def interact(
            self,
            state_features: np.ndarray,
            actions: List[Action]
    ) -> np.ndarray:
        """
        Interact a state-feature matrix with one-hot encoded action vectors.

        :param state_features: Feature matrix (#states, #features)
        :param actions: Actions.
        :return: State-action interacted feature matrix (#actions * #states, #actions * #features)
        """

        num_states = state_features.shape[0]
        num_actions = len(actions)
        if num_states != num_actions:
            raise ValueError(f'Expected {num_states} actions, but got {num_actions}')

        encoded_actions = self.action_encoder.transform(np.array(actions).reshape(-1, 1)).toarray()

        # interact each feature-vector with its associated one-hot encoded action vector
        interacted_state_features = np.array([
            [a * d for a, d in product(encoded_action, state_features_vector)]
            for state_features_vector, encoded_action in zip(state_features, encoded_actions)
        ])

        return interacted_state_features

    def __init__(
            self,
            environment: MdpEnvironment,
            actions: List[Action]
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        :param actions: Actions.
        """

        super().__init__(
            environment=environment
        )

        self.actions = actions

        # initialize the one-hot action encoder
        action_array = np.array([actions])
        self.action_encoder = OneHotEncoder(categories=action_array)
        self.action_encoder.fit(action_array.reshape(-1, 1))


@rl_text(chapter='Feature Extractors', page=1)
class StateActionIdentityFeatureExtractor(FeatureExtractor):
    """
    Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment. See
    `rlai.value_estimation.function_approximation.statistical_learning.feature_extraction.gridworld.GridworldFeatureExtractor`
    for an example.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        fex = StateActionIdentityFeatureExtractor(
            environment=environment
        )

        return fex, unparsed_args

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            for_fitting: bool
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param for_fitting: Whether the extracted features will be used for fitting (True) or prediction (False).
        :return: State-feature pandas.DataFrame or numpy.ndarray.
        """

        self.check_state_and_action_lists(states, actions)

        return pd.DataFrame([
            (state.i, action.i)
            for state, action in zip(states, actions)
        ], columns=self.get_feature_names())

    def get_feature_names(
            self
    ) -> List[str]:
        """
        Get names of extracted features.

        :return: List of feature names.
        """
        return ['s', 'a']

    def __init__(
            self,
            environment: MdpEnvironment
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        super().__init__(
            environment=environment
        )
