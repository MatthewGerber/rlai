class Action:
    """
    Base class for all actions.
    """

    def __init__(
            self,
            i: int,
            name: str = None
    ):
        """
        Initialize the action.

        :param i: Identifier for the action.
        :param name: Name (optional).
        """

        self.i = i
        self.name = name

    def __str__(
            self
    ) -> str:
        """
        Get string description of action.

        :return: String.
        """
        return f'{self.i}:  {self.name}'

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for action.

        :return: Hash code.
        """

        return self.i

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

        return self.i != other.i
