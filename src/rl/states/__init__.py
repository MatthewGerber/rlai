class State:

    def __init__(
            self,
            i: int
    ):
        """
        Initialize the state.

        :param i: Identifier for the state.
        """

        self.i = i
        self.hash = i.__hash__()

    def __str__(
            self
    ) -> str:
        """
        Get string description of state.

        :return: String.
        """
        return str(self.i)

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for state.

        :return: Hash code
        """

        return self.hash

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

        return not self.__eq__(other)
