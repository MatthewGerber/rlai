from typing import Dict, List, Optional, Union, Iterator

import numpy as np

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState


@rl_text(chapter=3, page=58)
class TabularPolicy(Policy):
    """
    Policy for use with tabular methods.
    """

    def get_state_i(
            self,
            state_descriptor: Union[str, np.ndarray]
    ) -> Optional[int]:
        """
        Get the integer identifier for a state. The returned value is guaranteed to be the same for the same state,
        both throughout the life of the current agent as well as after the current agent has been pickled for later
        use (e.g., in checkpoint-based resumption).

        :param state_descriptor: State descriptor, either a string (for discrete states) or an array representing a
        position within an n-dimensional continuous state space, which will be discretized.
        :return: Integer identifier.
        """

        if isinstance(state_descriptor, np.ndarray):

            if self.continuous_state_discretization_resolution is None:
                raise ValueError('Attempted to discretize a continuous state without a resolution.')

            state_descriptor = '|'.join(
                str(int(state_dim_value / self.continuous_state_discretization_resolution))
                for state_dim_value in state_descriptor
            )

        elif not isinstance(state_descriptor, str):
            raise ValueError(f'Unknown state space type:  {type(state_descriptor)}')

        if state_descriptor not in self.state_id_str_int:
            self.state_id_str_int[state_descriptor] = len(self.state_id_str_int)

        return self.state_id_str_int[state_descriptor]

    def __init__(
            self,
            continuous_state_discretization_resolution: Optional[float],
            SS: Optional[List[MdpState]]
    ):
        """
        Initialize the policy.

        :param continuous_state_discretization_resolution: Discretization resolution for continuous state spaces.
        :param SS: List of states for which to initialize the policy to be equiprobable over actions. This is useful for
        environments in which the list of states can be easily enumerated. It is not useful for environments (e.g.,
        `rlai.environments.mancala.Mancala`) in which the list of states is very large and difficult enumerate ahead of
        time. The latter problems should be addressed with a lazy-expanding list of states (see Mancala for an example).
        In such cases, pass None here.
        """

        if SS is None:
            SS = []

        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution

        self.state_action_prob: Dict[MdpState, Dict[Action, float]] = {
            s: {
                a: 1 / len(s.AA)
                for a in s.AA
            }
            for s in SS
        }

        self.state_id_str_int: Dict[str, int] = {}

    def __len__(
            self
    ) -> int:
        """
        Get the number of states in the policy.

        :return: Number of states.
        """

        return len(self.state_action_prob)

    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

        return state in self.state_action_prob

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        # if the policy is not defined for the state, then update the policy to be uniform across feasible actions.
        if state not in self.state_action_prob:
            self.state_action_prob[state] = {
                a: 1 / len(state.AA)
                for a in state.AA
            }

        return self.state_action_prob[state]

    def __iter__(
            self
    ) -> Iterator:
        """
        Get an iterator over the policies states and their action-probability dictionaries.
        :return: Iterator.
        """

        return iter(self.state_action_prob)

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current policy equals another.

        :param other: Other policy.
        :return: True if equal and False otherwise.
        """

        other: TabularPolicy

        return self.state_action_prob == other.state_action_prob

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the current policy does not equal another.

        :param other: Other policy.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
