from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import List, Tuple, Union, Dict, Optional

import numpy as np
import pandas as pd

from rlai.core import Action, MdpState
from rlai.core.environments.mdp import MdpEnvironment
from rlai.docs import rl_text
from rlai.models.feature_extraction import FeatureExtractor, OneHotCategoricalFeatureInteracter
from rlai.utils import parse_arguments


@rl_text(chapter=9, page=197)
class StateActionFeatureExtractor(FeatureExtractor):
    """
    Feature extractor.
    """

    @abstractmethod
    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            refit_scaler: bool
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature pandas.DataFrame, numpy.ndarray. A DataFrame is only valid with Patsy-style formula
        designs.
        """

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

    def get_action_feature_names(
            self
    ) -> Optional[Dict[str, List[str]]]:
        """
        Get names of actions and their associated feature names.

        :return: Dictionary of action names and their associated feature names.
        """

        return None

    def __init__(
            self,
            environment: MdpEnvironment,
            scale_features: bool
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        :param scale_features: Whether to scale features.
        """

        super().__init__()

        self.scale_features = scale_features


@rl_text(chapter='Feature Extractors', page=1)
class StateActionInteractionFeatureExtractor(StateActionFeatureExtractor, ABC):
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
            actions: List[Action],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Interact a state-feature matrix with one-hot encoded actions.

        :param state_features: Feature matrix (#states, #features)
        :param actions: Actions, with length equal to #states.
        :param refit_scaler: Whether to refit the scaler.
        :return: State-action interacted feature matrix (#states, #action levels * #features)
        """

        return self.interacter.interact(state_features, actions, refit_scaler)

    def __init__(
            self,
            environment: MdpEnvironment,
            actions: List[Action],
            scale_features: bool
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        :param actions: Actions.
        :param scale_features: Whether to scale features.
        """

        super().__init__(
            environment=environment,
            scale_features=scale_features
        )

        self.actions = actions

        self.interacter = OneHotCategoricalFeatureInteracter(self.actions, scale_features)


@rl_text(chapter='Feature Extractors', page=1)
class StateActionIdentityFeatureExtractor(StateActionFeatureExtractor):
    """
    Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment. See
    `rlai.gpi.state_action_value.function_approximation.statistical_learning.feature_extraction.gridworld.GridworldFeatureExtractor`
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
    ) -> Tuple[StateActionFeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls(
            environment=environment
        )

        return fex, unparsed_args

    def extracts_intercept(
            self
    ) -> bool:
        """
        Whether the feature extractor extracts an intercept (constant) term.

        :return: True if an intercept (constant) term is extracted and False otherwise.
        """

        return False

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            refit_scaler: bool
    ) -> pd.DataFrame:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature pandas.DataFrame.
        """

        self.check_state_and_action_lists(states, actions)

        return pd.DataFrame([
            (state.i, action.i)
            for state, action in zip(states, actions)
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
            environment=environment,
            scale_features=False
        )
