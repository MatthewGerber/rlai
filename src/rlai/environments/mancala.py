from argparse import ArgumentParser, Namespace
from typing import Tuple, List, Optional

from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.agents.mdp import Human, StochasticMdpAgent, MdpAgent
from rlai.environments import Environment
from rlai.environments.mdp import MdpEnvironment
from rlai.rewards import Reward
from rlai.states import State
from rlai.states.mdp import MdpState


class MancalaState(MdpState):
    """
    State of the mancala game. In charge of representing the entirety of the game state and advancing to the next state.
    """

    @staticmethod
    def player_1_is_next(
            picked_pocket,
            go_again: bool
    ) -> bool:
        """
        Gets whether or not it is player 1's turn next, based on the pocket that was just picked and whether sowing from
        that pocket has earned another turn.

        :param picked_pocket: Pocket that was picked.
        :param go_again: Whether or not sowing from the picked pocket earned another turn.
        :return: True if player 1 is next and False otherwise.
        """

        return (picked_pocket.player_1 and go_again) or (not picked_pocket.player_1 and not go_again)

    def advance(
            self,
            environment: Environment,
            t: int,
            a: Action,
            agent: MdpAgent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance from the current state given an action.

        :param environment: Environment.
        :param t: Current time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of next state and next reward.
        """

        environment: Mancala

        # pick and sow from pocket
        picked_pocket = environment.board[a.i]
        go_again = environment.sow_and_capture(picked_pocket)
        next_state = MancalaState(
            mancala=environment,
            player_1=MancalaState.player_1_is_next(picked_pocket, go_again),
            agent=agent
        )

        # check for termination
        next_reward = environment.r_none
        if next_state.terminal:
            next_reward = environment.get_terminal_reward()

        # if the agent (player 1) does not get to go again, let the environmental agent take its turn(s)
        elif not go_again:

            while True:

                environment.player_2.sense(next_state, t)

                # if the environmental agent is human, then render the board for them to see.
                if isinstance(environment.player_2, Human):
                    environment.render()

                p2_a = environment.player_2.act(t)
                picked_pocket = environment.board[p2_a.i]
                go_again = environment.sow_and_capture(picked_pocket)
                next_state = MancalaState(
                    mancala=environment,
                    player_1=MancalaState.player_1_is_next(picked_pocket, go_again),
                    agent=agent
                )

                # check for termination
                if next_state.terminal:
                    next_reward = environment.get_terminal_reward()
                    break

                # take another turn if earned
                elif not go_again:
                    break

        return next_state, next_reward

    def __init__(
            self,
            mancala,
            player_1: bool,
            agent: MdpAgent
    ):
        """
        Initialize the state.

        :param mancala: Mancala environment
        :param player_1: First player (agent).
        :param agent: Agent.
        """

        mancala: Mancala

        super().__init__(
            i=agent.get_state_i('-'.join(str(pit.count) for pit in mancala.board)),
            AA=mancala.get_feasible_actions(player_1),
            terminal=mancala.is_terminal()
        )


class Pit:
    """
    A general pit, either a pocket (something a player can pick from) or a store (the goal pit).
    """

    def pick(
            self
    ) -> int:
        """
        Pick all seeds from the current pit and return them.

        :return: Number of seeds picked.
        """

        count = self.count
        self.count = 0

        return count

    def sow(
            self,
            count: int
    ):
        """
        Sow (deposit) seeds into the current pit.

        :param count: Number of seeds.
        """

        self.count += count

    def __init__(
            self,
            player_1: bool,
            count: int,
            store: bool
    ):
        """
        Initialize the pit.

        :param player_1: Whether or not the current pit is for player 1.
        :param count: Number of seeds.
        :param store: Whether or not the current pit is a goal pit.
        """

        self.player_1 = player_1
        self.count = count
        self.store = store

        # these will be assigned after all pits have been created
        self.i = None
        self.opposing_pocket = None
        self.action = None

    def __str__(
            self
    ) -> str:
        """
        Override for `str` function.
        :return: String.
        """

        return f'{self.i}:  Player {1 if self.player_1 else 2}, {self.count}{"*" if self.store else ""}'


class Mancala(MdpEnvironment):
    """
    Environment for the mancala game.
    """

    @staticmethod
    def human_player_mutator(
            environment,
            **kwargs
    ):
        """
        Change the Mancala environment to let a human play the trained agent.

        :param environment: Environment.
        :param kwargs: Unused args.
        """

        environment: Mancala
        environment.player_2 = Human()

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parsed_args, unparsed_args = super().parse_arguments(args)

        parser = ArgumentParser(allow_abbrev=False)

        parser.add_argument(
            '--initial-count',
            type=int,
            help='Initial seed count in each pit.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        mancala = Mancala(
            random_state=random_state,
            player_2=StochasticMdpAgent(
                'environmental agent',
                random_state,
                None,
                1
            ),
            **vars(parsed_args)
        )

        return mancala, unparsed_args

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> State:
        """
        Reset the game to the initial state.

        :param agent: Agent.
        """

        super().reset_for_new_run(agent)

        for pocket in self.board:
            if pocket.store:
                pocket.count = 0
            else:
                pocket.count = self.initial_count

        self.state = MancalaState(
            mancala=self,
            player_1=True,
            agent=agent
        )

        return self.state

    def render(
            self
    ):
        """
        Render the board for interaction by a human agent.
        """

        print(f'|{self.player_1_store.count}', end='')
        for player_1_pocket in reversed(self.player_1_pockets):
            print(f'|{player_1_pocket.count}', end='')
        print('|')

        print('  ', end='')
        for player_2_pocket in self.player_2_pockets:
            print(f'|{player_2_pocket.count}', end='')

        print(f'|{self.player_2_store.count}|')

        print('  ', end='')
        for i, player_2_pocket in enumerate(self.player_2_pockets):
            print(f' {i}', end='')

        print()

    def get_terminal_reward(
            self
    ) -> Reward:
        """
        Get terminal reward.

        :return: Rewards for win, loss, and draw.
        """

        if self.player_1_store.count > self.player_2_store.count:
            reward = self.r_win
        elif self.player_2_store.count > self.player_1_store.count:
            reward = self.r_lose
        else:
            reward = self.r_none

        return reward

    def get_feasible_actions(
            self,
            player_1: Optional[bool]
    ) -> List[Action]:
        """
        Get actions that are feasible for a player to take given the current state of the board.

        :param player_1: Whether or not to check actions for player 1.
        :return: List of feasible actions.
        """

        if player_1:
            actions = [p.action for p in self.player_1_pockets if p.count > 0]
        else:
            actions = [p.action for p in self.player_2_pockets if p.count > 0]

        return actions

    def is_terminal(
            self
    ) -> bool:
        """
        Check whether the board is currently in a terminal state.

        :return: True for terminal and False otherwise.
        """

        return all(p.count == 0 for p in self.player_1_pockets) or all(p.count == 0 for p in self.player_2_pockets)

    def sow_and_capture(
            self,
            pocket
    ) -> bool:
        """
        Pick and sow seeds from a pocket.

        :param pocket: Pocket to pick from.
        :return: True if player gets to go again, based on where sown seeds landed.
        """

        # pick pocket
        pick_count = pocket.pick()
        if pick_count <= 0:
            raise ValueError('Cannot pick empty pocket.')

        # sow
        sow_pocket = None
        sow_i = pocket.i + 1
        while pick_count > 0:

            if sow_i >= len(self.board):
                sow_i = 0

            sow_pocket = self.board[sow_i]

            if sow_pocket.store and sow_pocket.player_1 != pocket.player_1:
                pass
            else:
                sow_pocket.sow(1)
                pick_count -= 1

            sow_i += 1

        go_again = False

        # go again if the final seed landed in the player's own store
        if sow_pocket.store and sow_pocket.player_1 == pocket.player_1:
            go_again = True
        # capture opponent's seeds if the final seed landed in one of the player's empty pits, and the opposing pit
        # contains seeds.
        elif sow_pocket.count == 1 and sow_pocket.player_1 == pocket.player_1 and sow_pocket.opposing_pocket.count > 0:
            to_store = sow_pocket.pick() + sow_pocket.opposing_pocket.pick()
            own_store = self.player_1_store if pocket.player_1 else self.player_2_store
            own_store.sow(to_store)

        return go_again

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            initial_count: int,
            player_2: Agent
    ):
        """
        Initialize the game.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param initial_count: Initial count for each pit.
        :param player_2: Agent for player 2.
        """

        super().__init__(
            name='mancala',
            random_state=random_state,
            T=T
        )

        self.initial_count = initial_count
        self.player_2 = player_2

        self.r_win = Reward(0, 1.0)
        self.r_lose = Reward(1, -1.0)
        self.r_none = Reward(2, 0.0)

        self.player_1_pockets = [
            Pit(True, self.initial_count, False)
            for _ in range(6)
        ]
        self.player_1_store = Pit(True, 0, True)

        self.player_2_pockets = [
            Pit(False, self.initial_count, False)
            for _ in range(6)
        ]
        self.player_2_store = Pit(False, 0, True)

        self.board = self.player_1_pockets + [self.player_1_store] + self.player_2_pockets + [self.player_2_store]

        for i, pit in enumerate(self.board):

            pit.i = i

            # non-store pit (i.e., pockets) have actions associated with them. Action.i indexes the particular pit
            # within the board.
            if not pit.store:
                pit.action = Action(pit.i)

        # Action.name indicates the i-th pit from the player's perspective
        for i, pit in enumerate(self.player_1_pockets):
            pit.action.name = str(i)

        for i, pit in enumerate(self.player_2_pockets):
            pit.action.name = str(i)

        for player_1_pocket, opposing_player_2_pocket in zip(self.player_1_pockets, reversed(self.player_2_pockets)):
            player_1_pocket.opposing_pocket = opposing_player_2_pocket
            opposing_player_2_pocket.opposing_pocket = player_1_pocket
