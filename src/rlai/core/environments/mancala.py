from argparse import ArgumentParser
from typing import Tuple, List, Optional, Union

from numpy.random import RandomState

from rlai.core import Reward, Action, MdpState, Agent, MdpAgent, Human, Environment
from rlai.core.environments.mdp import MdpEnvironment
from rlai.docs import rl_text
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator
from rlai.utils import parse_arguments


@rl_text(chapter='States', page=1)
class MancalaState(MdpState):
    """
    State of the mancala game. In charge of representing the entirety of the game state and advancing to the next state.
    """

    def __init__(
            self,
            mancala: 'Mancala',
            agent_to_sense_state: Union[MdpAgent, Human],
            truncated: bool
    ):
        """
        Initialize the state.

        :param mancala: Mancala environment
        :param agent_to_sense_state: Agent that will sense the state.
        :param truncated: Whether the state is truncated, meaning the episode has ended for some reason other than the
        natural dynamics of the environment. For example, imposing an artificial time limit on an episode might cause
        the episode to end without the agent in a predefined goal state.
        """

        agent_is_player_1 = agent_to_sense_state != mancala.player_2

        # get the pits ordered from the perspective of the player that will sense the resulting state
        if agent_is_player_1:
            state_pits = mancala.player_1_pockets + [mancala.player_1_store] + mancala.player_2_pockets + [mancala.player_2_store]
        else:
            state_pits = mancala.player_2_pockets + [mancala.player_2_store] + mancala.player_1_pockets + [mancala.player_1_store]

        # get state index from the agent that will sense the state
        state_i_str = '|'.join(str(pit.count) for pit in state_pits)
        state_i = agent_to_sense_state.pi.get_state_i(state_i_str)

        super().__init__(
            i=state_i,
            AA=mancala.get_feasible_actions(agent_is_player_1),
            terminal=mancala.is_terminal(),
            truncated=truncated
        )


class Pit:
    """
    A general pit, either a pocket (something a player can pick from) or a store (the goal pit).
    """

    def __init__(
            self,
            player_1: bool,
            count: int,
            store: bool
    ):
        """
        Initialize the pit.

        :param player_1: Whether the current pit is for player 1.
        :param count: Number of seeds.
        :param store: Whether the current pit is a goal pit.
        """

        self.player_1 = player_1
        self.count = count
        self.store = store

        # these will be assigned after all pits have been created
        self.i: Optional[int] = None
        self.opposing_pocket: Optional[Pit] = None
        self.action: Optional[Action] = None

    def __str__(
            self
    ) -> str:
        """
        Override for `str` function.

        :return: String.
        """

        return f'{self.i}:  Player {1 if self.player_1 else 2}, {self.count}{"*" if self.store else ""}'

    def pick(
            self
    ) -> int:
        """
        Pick all seeds from the current pit and return them.

        :return: Number of seeds picked.
        """

        if self.count <= 0:
            raise ValueError('Cannot pick empty pocket.')

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


@rl_text(chapter='Environments', page=1)
class Mancala(MdpEnvironment):
    """
    Environment for the mancala game. This is a simple game with many rule variations, and it provides a greater
    challenge in terms of implementation and state-space size than the gridworld. I have implemented a fairly common
    variation summarized below.

    * One row of 6 pockets per player, each starting with 4 seeds.
    * Landing in the store earns another turn.
    * Landing in own empty pocket steals.
    * Game terminates when a player's pockets are clear.
    * Winner determined by store count.

    A couple of hours of Monte Carlo optimization explores more than 1 million states when playing against an
    equiprobable random opponent.
    """

    @staticmethod
    def human_player_mutator(
            environment: 'Mancala',
            **_
    ):
        """
        Change the Mancala environment to let a human play the trained agent.

        :param environment: Environment.
        """

        environment.player_2 = Human()

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--initial-count',
            type=int,
            help='Initial seed count in each pit.'
        )

        return parser

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

        parsed_args, unparsed_args = parse_arguments(cls, args)

        mancala = cls(
            random_state=random_state,
            player_2=None,
            **vars(parsed_args)
        )

        return mancala, unparsed_args

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance from the current state given an action.

        :param state: State to advance.
        :param t: Current time step.
        :param a: Action.
        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: 2-tuple of next state and next reward.
        """

        assert isinstance(agent, MdpAgent)

        # pick and sow from pocket
        picked_pocket = self.board[a.i]
        go_again = self.sow_and_capture(picked_pocket)
        self.state = MancalaState(
            mancala=self,
            agent_to_sense_state=agent if go_again else self.player_2,
            truncated=self.T is not None and t >= self.T
        )

        # check for termination
        next_reward = self.r_none
        if self.state.terminal:
            next_reward = self.get_terminal_reward()

        # if the agent (player 1) does not get to go again, let the environmental agent take its turn(s)
        elif not go_again:

            while True:

                self.player_2.sense(self.state, t)

                # if the environmental agent is human, then render the board for them to see.
                if isinstance(self.player_2, Human):
                    self.render()

                p2_a = self.player_2.act(t)
                picked_pocket = self.board[p2_a.i]
                go_again = self.sow_and_capture(picked_pocket)
                self.state = MancalaState(
                    mancala=self,
                    agent_to_sense_state=self.player_2 if go_again else agent,
                    truncated=self.T is not None and t >= self.T
                )

                # check for termination
                if self.state.terminal:
                    next_reward = self.get_terminal_reward()
                    break

                # take another turn if earned
                elif not go_again:
                    break

        return self.state, next_reward

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> MdpState:
        """
        Reset the game to the initial state.

        :param agent: Agent used to generate on-the-fly state identifiers.
        """

        super().reset_for_new_run(agent)

        for pocket in self.board:
            if pocket.store:
                pocket.count = 0
            else:
                pocket.count = self.initial_count

        self.state = MancalaState(
            mancala=self,
            agent_to_sense_state=agent,
            truncated=False
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

        :param player_1: Whether to check actions for player 1.
        :return: List of feasible actions.
        """

        if player_1:
            actions = [p.action for p in self.player_1_pockets if p.count > 0]
        else:
            actions = [p.action for p in self.player_2_pockets if p.count > 0]

        return actions  # type: ignore[return-value]

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
            pocket: 'Pit'
    ) -> bool:
        """
        Pick and sow seeds from a pocket.

        :param pocket: Pocket to pick from.
        :return: True if player gets to go again, based on where sown seeds landed.
        """

        assert pocket.i is not None

        # pick pocket
        pick_count = pocket.pick()

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

        assert sow_pocket is not None

        go_again = False

        # go again if the final seed landed in the player's own store
        if sow_pocket.store and sow_pocket.player_1 == pocket.player_1:
            go_again = True

        # capture opponent's seeds if the final seed landed in one of the player's empty pits, and the opposing pit
        # contains seeds.
        elif (
            sow_pocket.count == 1 and
            sow_pocket.player_1 == pocket.player_1 and
            sow_pocket.opposing_pocket is not None and
            sow_pocket.opposing_pocket.count > 0
        ):
            to_store = sow_pocket.pick() + sow_pocket.opposing_pocket.pick()
            own_store = self.player_1_store if pocket.player_1 else self.player_2_store
            own_store.sow(to_store)

        return go_again

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            initial_count: int,
            player_2: Optional[Union[MdpAgent, Human]]
    ):
        """
        Initialize the game.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param initial_count: Initial count for each pit.
        :param player_2: Agent for player 2, or None to use a random agent.
        """

        super().__init__(
            name='mancala',
            random_state=random_state,
            T=T
        )

        if player_2 is None:
            player_2 = ActionValueMdpAgent(
                'environmental agent',
                random_state,
                1,
                TabularStateActionValueEstimator(self, None, None)
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

            # non-store pit (i.e., pockets) have actions associated with them. `Action.i` indexes the particular pit
            # within the board.
            if not pit.store:
                pit.action = Action(pit.i)

        # Action.name indicates the i-th pit from the player's perspective
        for i, pit in enumerate(self.player_1_pockets):
            assert pit.action is not None
            pit.action.name = str(i)

        for i, pit in enumerate(self.player_2_pockets):
            assert pit.action is not None
            pit.action.name = str(i)

        for player_1_pocket, opposing_player_2_pocket in zip(self.player_1_pockets, reversed(self.player_2_pockets)):
            player_1_pocket.opposing_pocket = opposing_player_2_pocket
            opposing_player_2_pocket.opposing_pocket = player_1_pocket
