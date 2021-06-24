from typing import Dict

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState
import numpy as np


@rl_text(chapter=13, page=321)
class ParameterizedPolicy(Policy):
    """
    Policy for use with policy gradient methods
    """

    def __init__(
            self,
            n_state_feature_dims: int
    ):
        """
        Initialize the parameterized policy.

        :param n_state_feature_dims: Number of dimensions in the state-action feature vector.
        """

        super().__init__()

        self.theta = np.zeros

    def __contains__(self, state: MdpState) -> bool:
        pass

    def __getitem__(self, state: MdpState) -> Dict[Action, float]:
        pass