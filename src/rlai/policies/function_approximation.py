from typing import Dict, Union

import numpy as np

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState


@rl_text(chapter=10, page=244)
class FunctionApproximationPolicy(Policy):
    """
    Policy for use with function approximation methods.
    """

    def get_state_i(
            self,
            state_descriptor: Union[str, np.ndarray]
    ) -> int:
        """
        Get the integer identifier for a state. The returned value is guaranteed to be the same for the same state,
        both throughout the life of the current agent as well as after the current agent has been pickled for later
        use (e.g., in checkpoint-based resumption).

        :param state_descriptor: State descriptor, either a string (for discrete states) or an array representing a
        position within an n-dimensional continuous state space.
        :return: Integer identifier.
        """
        pass

    def __init__(
            self
    ):
        pass

    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

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
        pass
