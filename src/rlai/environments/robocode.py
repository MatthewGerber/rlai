from typing import List, Union, Tuple, Dict, Any, Optional

import numpy as np
from flask import request
from numpy.random import RandomState

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.environments.rest import RestMdpEnvironment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.value_estimation.function_approximation.models import StateActionInteractionFeatureExtractor


@rl_text(chapter='Environments', page=1)
class RobocodeEnvironment(RestMdpEnvironment):
    """
    Robocode environment, provided by the Java implementation via calls to REST endpoints served by the inherited
    `RestMdpEnvironment`.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[MdpEnvironment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        return RobocodeEnvironment(
            name='robocode',
            random_state=random_state,
            T=None
        ), args

    def init_state_from_rest_put_dict(
            self,
            rest_request_dict: Dict[Any, Any]
    ) -> Tuple[MdpState, Reward]:
        """
        Initialize a state from the dictionary provided by the REST PUT (e.g., for setting and resetting the state).

        :param rest_request_dict: REST PUT dictionary.
        :return: 2-tuple of the state and reward.
        """

        dead = request.json['dead']
        win = request.json['win']
        terminal = dead or win

        state = RobocodeState(
            **rest_request_dict,
            actions=self.robot_actions,
            terminal=terminal
        )

        reward = Reward(
            i=None,
            r=1.0 if win else 0.0
        )

        return state, reward

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

        self.robot_actions = [
            RobocodeAction(0, 'ahead', 50.0),
            RobocodeAction(1, 'back', 50.0),
            RobocodeAction(2, 'fire', 1.0)
        ]


@rl_text(chapter='States', page=1)
class RobocodeState(MdpState):
    """
    State of Robocode battle.
    """

    def __init__(
            self,
            x: float,
            y: float,
            scanned_robot_event: Dict[Any, Any],
            dead: bool,
            win: bool,
            actions: List[Action],
            terminal: bool
    ):
        """
        Initialize the state.

        :param x: Robot x position.
        :param y: Robot y position.
        :param scanned_robot_event: Event information for scanning of another robot.
        :param dead: Whether or not the robot is dead.
        :param win: Whether or not the robot has won.
        :param actions: List of actions that can be taken.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=None,
            AA=actions,
            terminal=terminal
        )

        self.x = x
        self.y = y
        self.scanned_robot_event = scanned_robot_event
        self.dead = dead
        self.win = win


@rl_text(chapter='Feature Extractors', page=1)
class RobocodeFeatureExtractor(StateActionInteractionFeatureExtractor):
    """
    Robocode feature extractor.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[StateActionInteractionFeatureExtractor, List[str]]:

        return RobocodeFeatureExtractor(environment, [Action(1), Action(2), Action(3)]), args

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            for_fitting: bool
    ) -> Union[np.ndarray]:

        return np.array([
            [1, 2, 3]
            for a in actions
        ])


@rl_text(chapter='Actions', page=1)
class RobocodeAction(Action):
    """
    Robocode action.
    """

    def __init__(
            self,
            i: int,
            name: str,
            value: float
    ):
        """
        Initialize the action.

        :param i: Identifier for the action.
        :param name: Name.
        :param value: Value.
        """

        super().__init__(
            i=i,
            name=name
        )

        self.value = value
