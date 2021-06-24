from typing import Dict

import numpy as np

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState
from rlai.value_estimation.function_approximation.models import FeatureExtractor


@rl_text(chapter=13, page=321)
class ParameterizedPolicy(Policy):
    """
    Policy for use with policy gradient methods
    """

    def grad(
            self,
            a: Action,
            s: MdpState
    ) -> np.ndarray:

        x_s_a = self.fex.extract([s], [a], True)

        soft_max_num = np.exp(self.theta.dot(x_s_a))
        grad_soft_max_num = None
        soft_max_den = None
        grad_soft_max_den = None

    def __init__(
            self,
            fex: FeatureExtractor
    ):
        """
        Initialize the parameterized policy.

        :param fex: Feature extractor.
        """

        super().__init__()

        self.fex = fex

        self.theta = np.zeros(sum(len(features) for features in fex.get_action_feature_names().values()))

    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

        if state is None:
            raise ValueError('Attempted to check for None in policy.')

        return True

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        action_prob = {}

        return action_prob
