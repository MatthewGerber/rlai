from copy import deepcopy
from typing import Tuple, List, Dict, Optional

from numpy.random import RandomState

from rl.actions import Action
from rl.agents import Agent
from rl.environments import Environment
from rl.environments.mdp import MdpEnvironment
from rl.rewards import Reward
from rl.states.mdp import MdpState


class MancalaState(MdpState):
    """
    State of the Mancala game.
    """

    def advance(
            self,
            a: Action,
            t: int,
            random_state: RandomState
    ) -> Tuple[MdpState, int, Reward]:
        """
        Advance from the current state given an action.

        :param a: Action.
        :param t: Current time step.
        :param random_state: Random state.
        :return: 3-tuple of next state, next time step, and reward.
        """

        picked_pocket = self.mancala.board[a.i]
        next_state, go_again = self.sow_and_capture(picked_pocket)
        t += 1

        reward = self.mancala.r_none
        if next_state.terminal:
            reward = self.mancala.get_terminal_reward()

        elif not go_again:
            while True:
                next_state.mancala.p2.sense(next_state, t)
                p2_a = next_state.mancala.p2.act(t)
                picked_pocket = next_state.mancala.board[p2_a.i]
                next_state, go_again = next_state.sow_and_capture(picked_pocket)
                t += 1

                if next_state.terminal:
                    reward = next_state.mancala.get_terminal_reward()
                    break
                elif not go_again:
                    break

        return next_state, t, reward

    def sow_and_capture(
            self,
            pocket
    ) -> Tuple[MdpState, bool]:

        # make deep copy of state, as we're about to change it.
        next_state = MancalaState(
            AA=[],
            mancala=deepcopy(self.mancala)
        )

        # pick pocket
        pocket = next_state.mancala.board[pocket.i]
        pick_count = pocket.pick()
        if pick_count <= 0:
            raise ValueError('Cannot pick empty pocket.')

        # sow
        sow_pocket = None
        sow_i = pocket.i + 1
        while pick_count > 0:

            if sow_i >= len(next_state.mancala.board):
                sow_i = 0

            sow_pocket = next_state.mancala.board[sow_i]

            if sow_pocket.store and sow_pocket.player_1 != pocket.player_1:
                pass
            else:
                sow_pocket.sow(1)
                pick_count -= 1

            sow_i += 1

        go_again = False
        if sow_pocket.store and sow_pocket.player_1 == pocket.player_1:
            go_again = True
        elif sow_pocket.count == 1 and sow_pocket.player_1 == pocket.player_1 and sow_pocket.opposing_pocket.count > 0:
            to_store = sow_pocket.pick() + sow_pocket.opposing_pocket.pick()
            own_store = next_state.mancala.p1_store if pocket.player_1 else next_state.mancala.p2_store
            own_store.sow(to_store)

        # configure next state for the appropriate player
        p1 = (pocket.player_1 and go_again) or (not pocket.player_1 and not go_again)
        next_state.configure_copy(p1)

        return next_state, go_again

    def configure_copy(
            self,
            p1: bool
    ):
        self.i = self.get_id()
        self.hash = self.i.__hash__()
        self.terminal = self.is_terminal()
        self.AA = self.mancala.get_feasible_actions(None)
        self.AA_set = set(self.AA)
        self.AA_p1 = self.mancala.get_feasible_actions(True)
        self.AA_p2 = self.mancala.get_feasible_actions(False)
        self.set_feasible_actions(p1)

    def get_id(
            self
    ) -> str:
        return '-'.join(str(s.count) for s in self.mancala.board)

    def is_terminal(
            self
    ) -> bool:
        return all(p.count == 0 for p in self.mancala.p1_pockets) or all(p.count == 0 for p in self.mancala.p2_pockets)

    def set_feasible_actions(
            self,
            p1: bool
    ):
        if p1:
            self.AA = self.AA_p1
        else:
            self.AA = self.AA_p2

        self.AA_set = set(self.AA)

    def __init__(
            self,
            AA: List[Action],
            mancala
    ):
        self.mancala: Mancala = mancala
        self.AA_p1 = None
        self.AA_p2 = None

        super().__init__(
            i=0,
            AA=[
                a
                for a in AA
                if self.mancala.p1_pockets[a.i].count > 0
            ],
            terminal=self.is_terminal()
        )


class Pit:

    def pick(
            self
    ) -> int:
        count = self.count
        self.count = 0
        return count

    def sow(
            self,
            count: int
    ):
        self.count += count

    def __init__(
            self,
            player_1: bool,
            count: int,
            store: bool
    ):
        self.player_1 = player_1
        self.count = count
        self.store = store
        self.i = None
        self.opposing_pocket = None

    def __str__(
            self
    ) -> str:
        return f'{self.i}:  Player {1 if self.player_1 else 2}, {self.count}{"*" if self.store else ""}'


class Mancala(MdpEnvironment):

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        pass

    def get_terminal_reward(
            self
    ) -> Reward:

        if self.p1_store.count > self.p2_store.count:
            reward = self.r_win
        elif self.p2_store.count > self.p1_store.count:
            reward = self.r_lose
        else:
            reward = self.r_none

        return reward

    def get_feasible_actions(
            self,
            p1: Optional[bool]
    ) -> List[Action]:

        pockets = []

        if p1 is None or p1:
            pockets.extend(self.p1_pockets)

        if p1 is None or not p1:
            pockets.extend(self.p2_pockets)

        return [Action(pit.i) for pit in pockets if pit.count > 0]

    def __init__(
            self,
            random_state: RandomState,
            p2: Agent
    ):
        """
        Initialize the game.

        :param random_state: Random state.
        :param p2: Agent for player 2.
        """

        self.p2 = p2

        self.r_win = Reward(0, 1.0)
        self.r_lose = Reward(1, -1.0)
        self.r_none = Reward(2, 0.0)

        RR = [self.r_win, self.r_lose, self.r_none]

        self.p1_pockets = [
            Pit(True, 4, False)
            for _ in range(6)
        ]
        self.p1_store = Pit(True, 0, True)

        self.p2_pockets = [
            Pit(False, 4, False)
            for _ in range(6)
        ]
        self.p2_store = Pit(False, 0, True)

        self.board = self.p1_pockets + [self.p1_store] + self.p2_pockets + [self.p2_store]

        for i, pit in enumerate(self.board):
            pit.i = i

        for p1_pocket, opposing_p2_pocket in zip(self.p1_pockets, reversed(self.p2_pockets)):
            p1_pocket.opposing_pocket = opposing_p2_pocket
            opposing_p2_pocket.opposing_pocket = p1_pocket

        AA = [
            Action(i)
            for i in range(len(self.p1_pockets))
        ]

        initial_state = MancalaState(
            AA=AA,
            mancala=self
        )

        initial_state.configure_copy(True)

        super().__init__(
            name='mancala',
            AA=AA,
            random_state=random_state,
            SS=[initial_state],
            RR=RR
        )
