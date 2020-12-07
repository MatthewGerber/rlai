from abc import ABC
from typing import List, Optional

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.states import State


@rl_text(chapter=3, page=47)
class MdpState(State, ABC):
    """
    State of an MDP.
    """

    def __init__(
            self,
            i: Optional[int],
            AA: List[Action],
            terminal: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions that can be taken from this state.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=i,
            AA=AA
        )

        self.terminal = terminal
