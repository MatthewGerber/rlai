from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from itertools import product
from typing import List, Union, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.states.mdp import MdpState


@rl_text(chapter=9, page=197)
class FeatureExtractor(ABC):
    """
    Feature extractor.
    """

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parser = ArgumentParser(allow_abbrev=False)

        # future arguments to be added here...

        parsed_args, unparsed_args = parser.parse_known_args(args)

        return parsed_args, unparsed_args

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
            state: MdpState,
            actions: List[Action]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features from a state and actions.

        :param state: State.
        :param actions: Actions.
        :return: DataFrame with one row per action and one column per feature.
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
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parsed_args, unparsed_args = super().parse_arguments(args)

        parser = ArgumentParser(allow_abbrev=False)

        # future arguments to be added here...

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    def interact(
            self,
            actions: List[Action],
            state_features: np.ndarray
    ) -> np.ndarray:
        """
        Interact one-hot action vectors with a feature matrix.

        :param actions: Actions to interact.
        :param state_features: Feature matrix (#features)
        :return: Interacted feature matrix (#actions, #actions * #features)
        """

        if len(state_features.shape) != 1:
            raise ValueError('Expected a one-dimensional state-feature vector.')

        # one-hot encode the actions
        action_array = np.array(actions).reshape(-1, 1)
        encoded_actions = self.action_encoder.transform(action_array).toarray()

        # interact each action with the features
        state_features = np.array([
            [a * d for a, d in product(encoded_action, state_features)]
            for encoded_action in encoded_actions
        ])

        return state_features

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
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parsed_args, unparsed_args = super().parse_arguments(args)

        parser = ArgumentParser(allow_abbrev=False)

        # future arguments to be added here...

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

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

        parsed_args, unparsed_args = cls.parse_arguments(args)

        fex = StateActionIdentityFeatureExtractor(
            environment=environment
        )

        return fex, unparsed_args

    def extract(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract discrete state and action identifiers.

        :param state: State.
        :param actions: Actions.
        :return: DataFrame with one row per action and two columns (one for the state and one for the action).
        """

        return pd.DataFrame([
            (state.i, action.i)
            for action in actions
        ], columns=['s', 'a'])

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
