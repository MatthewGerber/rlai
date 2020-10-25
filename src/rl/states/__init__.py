from typing import Optional


class State:

    def __init__(
            self,
            i: Optional[int]
    ):
        """
        Initialize the state.

        :param i: Identifier for the state.
        """

        self.i = i

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
