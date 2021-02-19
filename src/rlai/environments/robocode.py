from typing import List, Union, Tuple

import numpy as np

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.states.mdp import MdpState
from rlai.value_estimation.function_approximation.models import StateActionInteractionFeatureExtractor


class RobocodeFeatureExtractor(StateActionInteractionFeatureExtractor):

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[StateActionInteractionFeatureExtractor, List[str]]:

        return RobocodeFeatureExtractor(environment, [Action(1), Action(2), Action(3)]), args

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            for_fitting: bool
    ) -> Union[np.ndarray]:

        return np.array([
            [1, 2, 3]
            for a in actions
        ])
