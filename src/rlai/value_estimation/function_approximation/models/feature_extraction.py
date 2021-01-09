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
            action_lists: List[List[Action]]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features for states and their associated actions.

        :param states: States.
        :param action_lists: Action lists, one list per state in `states`.
        :return: State-feature matrix.
        """
        pass

    @staticmethod
    def check_states_and_action_lists(
            states: List[MdpState],
            action_lists: List[List[Action]]
    ):
        """
        Check lengths of the state and action list lists.

        :param states: States.
        :param action_lists: Action lists.
        """

        num_states = len(states)
        num_action_lists = len(action_lists)
        if num_states != num_action_lists:
            raise ValueError(f'Expected {num_states} action lists but got {num_action_lists}')

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
            action_lists: List[List[Action]],
            state_features: np.ndarray
    ) -> np.ndarray:
        """
        Interact one-hot action vectors with a state-feature matrix.

        :param action_lists: Action lists (one list per row of `state_features`.
        :param state_features: Feature matrix (#states, #features)
        :return: Interacted feature matrix (#actions * #states, #actions * #features)
        """

        num_action_lists = len(action_lists)
        num_states = state_features.shape[0]
        if num_action_lists != num_states:
            raise ValueError(f'Expected {num_states} action lists, but got {num_action_lists}')

        # interact each one-hot encoded action with the features
        interacted_state_features = np.array([
            [a * d for a, d in product(encoded_action, state_features_row)]
            for actions, state_features_row in zip(action_lists, state_features)
            for encoded_action in self.action_encoder.transform(np.array(actions).reshape(-1, 1)).toarray()
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
            action_lists: List[List[Action]]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features for states and their associated actions.

        :param states: States.
        :param action_lists: Action lists, one list per state in `states`.
        :return: State-feature matrix.
        """

        self.check_states_and_action_lists(states, action_lists)

        return pd.DataFrame([
            (state.i, action.i)
            for state, action_list in zip(states, action_lists)
            for action in action_list
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
