import threading
from functools import partial
from typing import List, Tuple, Optional

from flask import Flask, jsonify
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.rewards import Reward
from rlai.states.mdp import MdpState


class RestMdpEnvironment(MdpEnvironment):

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[MdpEnvironment, List[str]]:

        return RestMdpEnvironment('test', random_state, None), args

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> MdpState:

        self.state_reward_event.wait()
        self.state_reward_event.clear()

        return self.state

    @staticmethod
    def get_action(
            self_ref
    ):
        self_ref.action_event.wait()
        self_ref.action_event.clear()

        return jsonify(
            action=self_ref.action.i
        )

    @staticmethod
    def set_state(
            self_ref
    ):
        self_ref.state = MdpState(1234, [Action(1), Action(2), Action(3)], False)
        self_ref.reward = Reward(None, 1.0)
        self_ref.state_reward_event.set()

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

        self.action: Optional[Action] = None
        self.action_event = threading.Event()
        self.action_event.clear()

        self.state: Optional[MdpState] = None
        self.reward: Optional[Reward] = None
        self.state_reward_event = threading.Event()
        self.state_reward_event.clear()

        self.flask_app = Flask(__name__)

        self.flask_app.add_url_rule(
            rule='/get-action',
            endpoint='get-action',
            view_func=partial(self.get_action, self_ref=self),
            methods=['GET']
        )

        self.flask_app.add_url_rule(
            rule='/set-state',
            endpoint='set-state',
            view_func=partial(self.set_state, self_ref=self),
            methods=['PUT']
        )

        self.flask_thread = threading.Thread(
            target=self.flask_app.run,
            kwargs={
                'port': 12345,
                'use_reloader': False
            }
        )

        self.flask_thread.start()
