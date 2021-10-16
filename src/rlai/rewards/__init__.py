from typing import Optional

from rlai.meta import rl_text


@rl_text(chapter='Rewards', page=1)
class Reward:
    """
    Reward.
    """

    def __init__(
            self,
            i: Optional[int],
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

        return f'Reward{"" if self.i is None else " " + str(self.i)}: {self.r}'

    def __hash__(
            self
    ) -> int:
        """
        Get hash code for reward.

        :return: Hash code
        """

        if self.i is None:
            raise ValueError('Cannot hash when i is None.')

        return self.i

    def __eq__(
            self,
            other: 'Reward'
    ) -> bool:
        """
        Check whether the current reward equals another.

        :param other: Other reward.
        :return: True if equal and False otherwise.
        """

        if self.i is None:
            raise ValueError('Cannot check equality when i is None.')

        return self.i == other.i

    def __ne__(
            self,
            other: 'Reward'
    ) -> bool:
        """
        Check whether the current reward is not equal to another.

        :param other: Other reward.
        :return: True if not equal and False otherwise.
        """

        if self.i is None:
            raise ValueError('Cannot check inequality when i is None.')

        return self.i != other.i
