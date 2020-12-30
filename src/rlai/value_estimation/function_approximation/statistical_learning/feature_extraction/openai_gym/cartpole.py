from argparse import ArgumentParser
from typing import List, Union, Tuple

import numpy as np
import pandas as pd
from gym.spaces import Discrete
from sklearn.preprocessing import PolynomialFeatures

from rlai.actions import Action
from rlai.environments.openai_gym import Gym, GymState
from rlai.meta import rl_text
from rlai.utils import parse_arguments
from rlai.value_estimation.function_approximation.statistical_learning.feature_extraction import (
    StateActionInteractionFeatureExtractor
)


@rl_text(chapter='Feature Extractors', page=1)
class CartpoleFeatureExtractor(StateActionInteractionFeatureExtractor):
    """
    A feature extractor for the OpenAI cartpole environment. This extractor, being based on the
    `StateActionInteractionFeatureExtractor`, directly extracts the fully interacted state-action feature matrix. It
    returns numpy.ndarray feature matrices, which are not compatible with the Patsy formula-based interface.
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
            environment: Gym
    ) -> Tuple[StateActionInteractionFeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        fex = CartpoleFeatureExtractor(
            environment=environment
        )

        return fex, unparsed_args

    def extract(
            self,
            state: GymState,
            actions: List[Action]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features.

        :param state: State.
        :param actions: Actions.
        :return: Feature matrix (#actions, #actions * #features)
        """

        state_features = self.polynomial_features.fit_transform(np.array([state.observation]))[0]

        return self.interact(
            actions=actions,
            state_features=state_features
        )

    def __init__(
            self,
            environment: Gym
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        if not isinstance(environment.gym_native.action_space, Discrete):
            raise ValueError('Expected a discrete action space, but did not get one.')

        super().__init__(
            environment=environment,
            actions=[
                Action(i)
                for i in range(environment.gym_native.action_space.n)
            ]
        )

        self.polynomial_features = PolynomialFeatures(
            degree=environment.gym_native.observation_space.shape[0],
            interaction_only=True,
            include_bias=False
        )
