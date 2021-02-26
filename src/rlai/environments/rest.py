import logging
import threading
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from functools import partial
from typing import Tuple, Optional, Any, Dict

from flask import Flask, jsonify, request, Response
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states.mdp import MdpState


@rl_text(chapter='Environments', page=1)
class RestMdpEnvironment(MdpEnvironment, ABC):
    """
    An MDP environment served up via calls to REST entpoints.
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
            help='Port to serve REST endpoints on.'
        )

        parser.add_argument(
            '--rest-verbose',
            action='store_true',
            help='Whether or not to print REST logging messages to console.'
        )

        return parser

    @staticmethod
    def rest_reset_for_new_run(
            self
    ) -> str:
        """
        Reset the environment for a new run.

        :param self: Reference to the environment. The current function is called by Flask with no arguments, which is
        why we set up a partial function, pass it the environment, and mark the current function static. If the
        function is not marked static, then the self instance and the partial argument both get passed.
        :return: Empty string.
        """

        self.state, _ = self.extract_state_and_reward_from_put_request(request.json)
        self.client_reset_for_new_run.set()

        return ""

    @staticmethod
    def rest_get_action(
            self
    ) -> Response:
        """
        Get the local environment's next action, to be used in advancing the calling environment.

        :param self: Reference to the environment. The current function is called by Flask with no arguments, which is
        why we set up a partial function, pass it the environment, and mark the current function static. If the
        function is not marked static, then the self instance and the partial argument both get passed.
        :return: Response containing the action to be used to advance the calling environment.
        """

        # wait for the server to set the action
        self.server_set_action.wait()
        self.server_set_action.clear()

        return jsonify(
            action=self.action.__dict__
        )

    @staticmethod
    def rest_set_state(
            self
    ) -> str:
        """
        Set the local environment's next state, which resulted from the action returned to the calling environment.

        :param self: Reference to the environment. The current function is called by Flask with no arguments, which is
        why we set up a partial function, pass it the environment, and mark the current function static. If the
        function is not marked static, then the self instance and the partial argument both get passed.
        :return: Empty string.
        """

        self.state, self.reward = self.extract_state_and_reward_from_put_request(request.json)
        self.client_set_state_and_reward.set()

        return ""

    @abstractmethod
    def extract_state_and_reward_from_put_request(
            self,
            rest_request_dict: Dict[Any, Any]
    ) -> Tuple[MdpState, Reward]:
        """
        Initialize a state from the dictionary provided by the REST PUT (e.g., for setting and resetting the state).

        :param rest_request_dict: REST PUT dictionary.
        :return: 2-tuple of the state and reward.
        """

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> MdpState:
        """
        Reset the environment for a new run.

        :param agent: Agent.
        :return: Initial state.
        """

        # wait for the client to reset
        self.client_reset_for_new_run.wait()
        self.client_reset_for_new_run.clear()

        return self.state

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance the simulation environment by setting the action for the client to retrieve and waiting for the client
        to send the updated state back.

        :param state: Current state.
        :param t: Current time step.
        :param a: Action for the client environment to take.
        :param agent: Agent.
        :return: Next state and reward.
        """

        # set action for client to retrieve and signal event
        self.action = a
        self.server_set_action.set()

        # wait for the client to reply
        self.client_set_state_and_reward.wait()
        self.client_set_state_and_reward.clear()

        return self.state, self.reward

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            port: int,
            rest_verbose: bool
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param port: Port to serve REST endpoints on.
        :param rest_verbose: Whether or not to print Flask logging messages to console.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.flask_app = Flask(__name__)

        # set up REST endpoints

        self.client_reset_for_new_run = threading.Event()
        self.client_reset_for_new_run.clear()
        self.flask_app.add_url_rule(
            rule='/reset-for-new-run',
            endpoint='reset-for-new-run',
            view_func=partial(self.rest_reset_for_new_run, self=self),
            methods=['PUT']
        )

        self.action: Optional[Action] = None
        self.server_set_action = threading.Event()
        self.server_set_action.clear()
        self.flask_app.add_url_rule(
            rule='/get-action',
            endpoint='get-action',
            view_func=partial(self.rest_get_action, self=self),
            methods=['GET']
        )

        self.state: Optional[MdpState] = None
        self.reward: Optional[Reward] = None
        self.client_set_state_and_reward = threading.Event()
        self.client_set_state_and_reward.clear()
        self.flask_app.add_url_rule(
            rule='/set-state',
            endpoint='set-state',
            view_func=partial(self.rest_set_state, self=self),
            methods=['PUT']
        )

        # run flask server in separate thread
        self.flask_thread = threading.Thread(
            target=self.flask_app.run,
            kwargs={
                'port': port,
                'use_reloader': False
            }
        )

        if not rest_verbose:
            log = logging.getLogger('werkzeug')
            log.disabled = True

        self.flask_thread.start()
