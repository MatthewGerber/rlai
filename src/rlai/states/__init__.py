from typing import Optional, List

from rlai.actions import Action


class State:

    def is_feasible(
            self,
            a: Action
    ) -> bool:
        """
        Check whether an action is feasible from the current state. This uses a set-based lookup with O(1) complexity,
        which is far faster than checking for the action in self.AA.

        :param a: Action.
        :return: True if the action is feasible from the current state and False otherwise.
        """

        return a in self.AA_set

    def __init__(
            self,
            i: Optional[int],
            AA: List[Action]
    ):
        """
        Initialize the state.

        :param i: Identifier for the state.
        :param AA: All actions that can be taken from this state.
        """

        self.i = i
        self.AA = AA

        # use set for fast existence checks (e.g., in `feasible` function)
        self.AA_set = set(self.AA)

    def __str__(
            self
    ) -> str:
        """
        Get string description of state.

        :return: String.
        """
        return f'State {self.i}'

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for state.

        :return: Hash code
        """

        return self.i

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current state equals another.

        :param other: Other state.
        :return: True if equal and False otherwise.
        """

        return self.i == other.i

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the current state is not equal to another.

        :param other: Other state.
        :return: True if not equal and False otherwise.
        """

        return self.i != other.i
