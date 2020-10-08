from typing import List, Dict

from rl.actions import Action
from rl.rewards import Reward
from rl.states import State


class MdpState(State):
    """
    MDP state.
    """

    def __init__(
            self,
            i: int,
            AA: List[Action],
            RR: List[Reward],
            terminal: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions that can be taken from this state.
        :param RR: All rewards provided by the environment
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=i
        )

        self.AA = AA
        self.RR = RR
        self.terminal = terminal
        self.SS = []

        # initialize an empty model within the state (see `init_model` for initialization)
        self.p_S_prime_R_given_A: Dict[
            Action, Dict[
                MdpState, Dict[
                    Reward, float
                ]
            ]
        ] = {}

    def init_model(
            self,
            SS: List[State]
    ):
        """
        Initialize the model within each state with zeros.

        :param SS: List of all states.
        """

        self.SS = SS

        self.p_S_prime_R_given_A = {
            a: {
                s: {
                    r: 0.0
                    for r in self.RR
                }
                for s in self.SS
            }
            for a in self.AA
        }
