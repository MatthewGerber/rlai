class Reward:

    def __init__(
            self,
            i: int,
            r: float
    ):
        """
        Initialize the reward.

        :param i: Identifier for the reward.
        :param r: Reward value.
        """

        self.i = i
        self.r = r

    def __str__(
            self
    ) -> str:
        """
        Get string description of reward.

        :return: String.
        """
        return f'Reward {self.i}: {self.r}'

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for reward.

        :return: Hash code
        """

        return self.i

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current reward equals another.

        :param other: Other reward.
        :return: True if equal and False otherwise.
        """

        return self.i == other.i

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the current reward is not equal to another.

        :param other: Other reward.
        :return: True if not equal and False otherwise.
        """

        return self.i != other.i
