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
            SS: List[State],
            RR: List[Reward],
            terminal: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions.
        :param SS: All states.
        :param RR: All rewards.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=i
        )

        self.AA = AA
        self.SS = SS
        self.RR = RR
        self.terminal = terminal

        # initialize an empty model within the state. can't fill it in until all states have been initialized.
        self.p_S_prime_R_given_A: Dict[
            Action, Dict[
                State, Dict[
                    Reward, float
                ]
            ]
        ] = {}

    def init_model(
            self
    ):
        """
        Initialize the model within each state with zeros.
        """

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
