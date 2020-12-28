from typing import Optional

import numpy as np


class Action:
    """
    Base class for all actions.
    """

    def __init__(
            self,
            i: int,
            name: Optional[str] = None
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


class DiscretizedAction(Action):
    """
    Action that is derived from discretizing an n-dimensional continuous action space.
    """

    def __init__(
            self,
            i: int,
            continuous_value: np.ndarray,
            name: Optional[str] = None
    ):
        """
        Initialize the action.

        :param i: Identifier for the action.
        :param continuous_value: Continuous n-dimensional action.
        :param name: Name (optional).
        """

        super().__init__(
            i=i,
            name=name
        )

        self.continuous_value = continuous_value
