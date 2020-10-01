from abc import ABC
from typing import List

from numpy.random import RandomState

from rl.actions import Action
from rl.environments import Environment
from rl.meta import rl_text
from rl.rewards import Reward
from rl.states import State


@rl_text(chapter=3, page=47)
class MDP(Environment, ABC):

    def __init__(
            self,
            name: str,
            AA: List[Action],
            random_state: RandomState,
            SS: List[State],
            RR: List[Reward]
    ):
        super().__init__(
            name=name,
            AA=AA,
            random_state=random_state
        )

        self.SS = SS
        self.RR = RR
