from abc import ABC


class Action(ABC):

    def __init__(
            self,
            i: int
    ):
        """
        Initialize the action.

        :param i: Identifier for the action.
        """

        self.i = i
        self.hash = i.__hash__()

    def __str__(
            self
    ) -> str:
        """
        Get string description of action.

        :return: String.
        """
        return str(self.i)

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for action.

        :return: Hash code
        """

        return self.hash

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current action equals another.

        :param other: Other action.
        :return: True if equal and False otherwise.
        """

        return self.i == other.i

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the current action is not equal to another.

        :param other: Other action.
        :return: True if not equal and False otherwise.
        """

        return not self.__eq__(other)
