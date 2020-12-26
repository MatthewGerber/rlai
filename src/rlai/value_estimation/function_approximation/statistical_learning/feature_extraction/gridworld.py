from typing import List, Union

import numpy as np
import pandas as pd

from rlai.actions import Action
from rlai.environments.mdp import Gridworld
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.value_estimation.function_approximation.statistical_learning.feature_extraction import (
    StateActionInteractionFeatureExtractor
)


@rl_text(chapter='Feature Extractors', page=1)
class GridworldFeatureExtractor(StateActionInteractionFeatureExtractor):
    """
    A feature extractor for the gridworld. This extractor, being based on the `StateActionInteractionFeatureExtractor`,
    directly extracts the fully interacted state-action feature matrix. It returns numpy.ndarray feature matrices, which
    are not compatible with the Patsy formula-based interface.
    """

    def extract(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features.

        :param state: State.
        :param actions: Actions.
        :return: Feature matrix (#actions, #actions * #features)
        """

        num_rows = self.environment.grid.shape[0]
        num_cols = self.environment.grid.shape[1]
        row = int(state.i / num_cols)
        col = state.i % num_cols

        state_features = np.array([
            row,  # from top
            num_rows - row - 1,  # from bottom
            col,  # from left
            num_cols - col - 1  # from right
        ])

        return self.interact(
            actions=actions,
            state_features=state_features
        )

    def __init__(
            self,
            environment: Gridworld
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        super().__init__(
            environment=environment,
            actions=[
                environment.a_up,
                environment.a_down,
                environment.a_left,
                environment.a_right
            ]
        )
