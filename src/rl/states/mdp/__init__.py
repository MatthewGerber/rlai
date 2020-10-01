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
            RR: List[Reward]
    ):
        super().__init__(
            i=i
        )

        self.p_S_prime_R_given_A: Dict[
            Action, Dict[
                State, Dict[
                    Reward, float
                ]
            ]
        ] = {
            a: {
                s: {
                    r: 0.0
                    for r in RR
                }
                for s in SS
            }
            for a in AA
        }
