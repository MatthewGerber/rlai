import logging
import threading
from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, Optional, Any, Dict

from flask import Flask, jsonify, request
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

    @staticmethod
    def rest_reset_for_new_run(
            self
    ) -> str:
        """
        Reset the environment for a new run.

        :param self: Reference to the environment. The current function is called by Flask with no arguments, which is
        why we set up a partial function, pass it the environment, and mark the current function static.
        :return: Empty string.
        """

        self.state = self.init_state_from_rest_request_dict(request.json, False)
        self.reset_event.set()

        return ""

    @abstractmethod
    def init_state_from_rest_request_dict(
            self,
            rest_request_dict: Dict[Any, Any],
            terminal: bool
    ) -> MdpState:
        """
        Initialize a state from its REST request dictionary.

        :param rest_request_dict: REST request dictionary.
        :param terminal: Whether or not the state is terminal.
        :return: State.
        """

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> MdpState:

        self.reset_event.wait()
        self.reset_event.clear()

        return self.state

    @staticmethod
    def rest_get_action(
            self
    ):
        self.action_event.wait()
        self.action_event.clear()

        return jsonify(
            action=self.action.name,
            value=1.0
        )

    @staticmethod
    def rest_set_state(
            self
    ):
        dead = request.json['dead']
        win = request.json['win']
        terminal = dead or win
        reward = 1.0 if win else 0.0

        self.state = self.init_state_from_rest_request_dict(request.json, terminal=terminal)
        self.reward = Reward(None, reward)
        self.state_reward_event.set()

        return ""

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:

        self.action = a
        self.action_event.set()

        self.state_reward_event.wait()
        self.state_reward_event.clear()

        return self.state, self.reward

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.flask_app = Flask(__name__)

        self.reset_event = threading.Event()
        self.reset_event.clear()
        self.flask_app.add_url_rule(
            rule='/reset-for-new-run',
            endpoint='reset-for-new-run',
            view_func=partial(self.rest_reset_for_new_run, self=self),
            methods=['PUT']
        )

        self.action: Optional[Action] = None
        self.action_event = threading.Event()
        self.action_event.clear()
        self.flask_app.add_url_rule(
            rule='/get-action',
            endpoint='get-action',
            view_func=partial(self.rest_get_action, self=self),
            methods=['GET']
        )

        self.state: Optional[MdpState] = None
        self.reward: Optional[Reward] = None
        self.state_reward_event = threading.Event()
        self.state_reward_event.clear()
        self.flask_app.add_url_rule(
            rule='/set-state',
            endpoint='set-state',
            view_func=partial(self.rest_set_state, self=self),
            methods=['PUT']
        )

        self.flask_thread = threading.Thread(
            target=self.flask_app.run,
            kwargs={
                'port': 12345,
                'use_reloader': False
            }
        )

        # log = logging.getLogger('werkzeug')
        # log.disabled = True

        self.flask_thread.start()
