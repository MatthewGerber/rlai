import json
import socket
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from typing import Tuple, Optional, Any, Dict

from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states.mdp import MdpState


@rl_text(chapter='Environments', page=1)
class TcpMdpEnvironment(MdpEnvironment, ABC):
    """
    An MDP environment served over a TCP connection.
    """

    @classmethod
    def get_argument_parser(
            cls,
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--port',
            type=int,
            default=54321,
            help='Port to serve environment on.'
        )

        return parser

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> MdpState:
        """
        Reset the environment for a new run.

        :param agent: Agent.
        :return: Initial state.
        """

        state_dict = json.loads(self.read_from_client())
        self.state, _ = self.extract_state_and_reward_from_client_dict(state_dict)

        return self.state

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance the simulation environment by writing an action to the client and waiting for the client to send the
        updated state back.

        :param state: Current state.
        :param t: Current time step.
        :param a: Action for the client environment to take.
        :param agent: Agent.
        :return: Next state and reward.
        """

        # write action to client
        self.write_to_client(json.dumps(a.__dict__))

        # read state/reward response
        next_state_dict = json.loads(self.read_from_client())
        self.state, reward = self.extract_state_and_reward_from_client_dict(next_state_dict)

        return self.state, reward

    @abstractmethod
    def extract_state_and_reward_from_client_dict(
            self,
            client_dict: Dict[Any, Any]
    ) -> Tuple[MdpState, Reward]:
        """
        Extract the state and reward from a client dict.

        :param client_dict: Client dictionary.
        :return: 2-tuple of the state and reward.
        """

    def read_from_client(
            self
    ) -> str:
        """
        Read a message from the client.

        :return: Message string.
        """

        return self.server_connection.recv(999999999).decode('utf-8')

    def write_to_client(
            self,
            s: str
    ):
        """
        Write a message to the client.

        :param s: Message string.
        """

        self.server_connection.sendall(f'{s}\n'.encode('utf-8'))

    def close(
            self
    ):
        """
        Close the environment.
        """

        self.server_socket.close()

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            port: int
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param port: Port to serve REST endpoints on.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(('127.0.0.1', port))

        print('Listening for client...', end='')
        self.server_socket.listen()
        self.server_connection, client_address = self.server_socket.accept()
        print(f'client connected:  {client_address}')
