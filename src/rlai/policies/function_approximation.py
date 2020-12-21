from typing import Dict, Union

import numpy as np

from rlai.actions import Action
from rlai.policies import Policy
from rlai.states.mdp import MdpState


class FunctionApproximationPolicy(Policy):

    def get_state_i(
            self,
            state_descriptor: Union[str, np.ndarray]
    ) -> int:
        pass

    def __init__(
            self
    ):
        pass

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        pass
