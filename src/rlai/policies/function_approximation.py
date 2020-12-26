from typing import Dict, List

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.states.mdp import MdpState


@rl_text(chapter=10, page=244)
class FunctionApproximationPolicy(Policy):
    """
    Policy for use with function approximation methods. This is effectively an interface to the underlying function
    approximation estimator and its reward model, which are accessed by indexing the policy with a state (e.g., a call
    like agent.pi[state]), which returns an action-probability dictionary.
    """

    def format_state_action_probs(
            self,
            states: List[MdpState]
    ) -> str:
        """
        Get a formatted string containing state-action probabilities for a list of states.

        :param states: States.
        :return: String.
        """

        s = ''
        for state in states:
            s += f'{state}\n'
            for action, prob in self[state].items():
                s += f'\tPr(A={action.name}):  {prob}\n'

        return s

    def format_state_action_values(
            self,
            states: List[MdpState]
    ) -> str:
        """
        Get a formatted string containing state-action values for a list of states.

        :param states: States.
        :return: String.
        """

        s = ''
        for state in states:
            s += f'{state}\n'
            for action, value in zip(state.AA, self.estimator.evaluate(state, state.AA)):
                s += f'\tq(S={state}, A={action.name}):  {value}\n'

        return s

    def __init__(
            self,
            estimator
    ):
        """
        Initialize the policy.

        :param estimator: State-action value estimator.
        """

        self.estimator = estimator

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
        Get action-probability dictionary for a state, accounting for the current value of epsilon that is stored in the
        estimator associated with this policy.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        values = self.estimator.evaluate(state, state.AA)
        max_value = max(values)
        num_maximizers = sum(value == max_value for value in values)
        action_prob = {
            action: (((1 - self.estimator.epsilon) / num_maximizers) if value == max_value else 0.0) + self.estimator.epsilon / len(values)
            for action, value in zip(state.AA, values)
        }

        return action_prob

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current function approximation policy equals another. Two such policies are equal if their
        associated estimators are equal.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """

        return self.estimator == other.estimator

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the current estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
