from copy import deepcopy
from typing import Tuple, List, Callable, Dict

from numpy.random import RandomState

from rl.actions import Action
from rl.agents import Agent
from rl.environments import Environment
from rl.environments.mdp import MdpEnvironment
from rl.rewards import Reward
from rl.states.mdp import MdpState


class MancalaState(MdpState):

    def advance(
            self,
            a: Action,
            t: int,
            random_state: RandomState
    ) -> Tuple[MdpState, Reward]:

        picked_pocket = self.mancala.p1_pockets[a.i]
        go_again = self.sow_and_capture(picked_pocket)
        next_state = self.state_generator()
        if next_state.terminal:
            return next_state, self.r_win

        if not go_again:
            while True:
                self.opponent.sense(next_state, t + 1)
                opponent_a = self.opponent.act(t + 1)
                picked_pocket = self.mancala.p2_pockets[opponent_a.i]
                go_again = self.sow_and_capture(picked_pocket)
                next_state = self.state_generator()
                if next_state.terminal:
                    return next_state, self.r_lose
                elif not go_again:
                    break

        return next_state, self.r_none

    def sow_and_capture(
            self,
            pocket
    ) -> bool:

        pocket: Pit

        pick_count = pocket.pick()
        if pick_count <= 0:
            raise ValueError('Cannot pick empty pocket.')

        sow_pocket = None
        sow_i = pocket.i + 1
        while pick_count > 0:

            if sow_i >= len(self.mancala.board):
                sow_i = 0

            sow_pocket = self.mancala.board[sow_i]

            if sow_pocket.store and sow_pocket.player_1 != pocket.player_1:
                continue

            sow_pocket.sow(1)
            sow_i += 1
            pick_count -= 1

        go_again = False

        if sow_pocket.store and sow_pocket.player_1 == pocket.player_1:
            go_again = True
        elif sow_pocket.count == 1 and sow_pocket.player_1 == pocket.player_1:
            go_again = True
            if sow_pocket.opposing.count > 0:
                stolen = sow_pocket.opposing.pick()
                own_store = self.mancala.p1_store if pocket.player_1 else self.mancala.p2_store
                own_store.sow(stolen)

        return go_again

    def get_id(
            self
    ) -> str:
        return '-'.join(str(s.count) for s in self.mancala.board)

    def is_terminal(
            self
    ) -> bool:
        return all(p.count == 0 for p in self.mancala.p1_pockets) or all(p == 0 for p in self.mancala.p2_pockets)

    def __init__(
            self,
            i: int,
            AA: List[Action],
            mancala,
            state_generator: Callable[[], MdpState]
    ):
        self.mancala: Mancala = mancala
        self.state_generator = state_generator

        super().__init__(
            i=i,
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
        self.opposing = None

    def __str__(
            self
    ) -> str:
        return f'{self.i}:  Player {1 if self.player_1 else 2}, {self.count}{"*" if self.store else ""}'


class Mancala(MdpEnvironment):

    id_state: Dict[str, MancalaState] = {}

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        pass

    @classmethod
    def get_state(
            cls,
            state: MancalaState
    ) -> MancalaState:

        state_id = state.get_id()

        if state_id not in cls.id_state:
            state.terminal = state.is_terminal()
            cls.id_state[state_id] = deepcopy(state)

        return cls.id_state[state_id]

    def __init__(
            self,
            random_state: RandomState,
            opponent: Agent
    ):
        """
        Initialize the game.

        :param random_state: Random state.
        :param opponent: Opponent agent.
        """

        self.opponent = opponent

        RR = [
            Reward(0, 1.0),
            Reward(1, -1.0),
            Reward(2, 0.0)
        ]

        self.r_win, self.r_lose, self.r_none = RR

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

        for pocket, opposing in zip(self.board, reversed(self.board)):
            pocket.opposing_pocket = opposing

        AA = [
            Action(i)
            for i in range(len(self.p1_pockets))
        ]

        SS = [
            MancalaState(
                i=0,
                AA=AA,
                mancala=self,
                state_generator=self.get_state
            )
        ]

        self.id_state = {
            s.get_id(): Mancala.get_state(s)
            for s in SS
        }

        super().__init__(
            name='mancala',
            AA=AA,
            random_state=random_state,
            SS=SS,
            RR=RR
        )
