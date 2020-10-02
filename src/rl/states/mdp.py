from typing import List, Dict

from rl.actions import Action
from rl.rewards import Reward
from rl.states import State


class MdpState(State):

    def __init__(
            self,
            i: int,
            AA: List[Action],
            SS: List[State],
            RR: List[Reward],
            terminal: bool
    ):
        super().__init__(
            i=i
        )

        self.AA = AA
        self.SS = SS
        self.RR = RR
        self.terminal = terminal

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
