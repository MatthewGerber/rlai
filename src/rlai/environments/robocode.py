from argparse import ArgumentParser
from typing import List, Union, Tuple, Dict, Any, Optional

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.environments.rest import RestMdpEnvironment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments
from rlai.value_estimation.function_approximation.models import StateActionInteractionFeatureExtractor


@rl_text(chapter='Environments', page=1)
class RobocodeEnvironment(RestMdpEnvironment):
    """
    Robocode environment binding to the official Java implementation. The Java implementation runs alongside the current
    environment, and a specialized robot implementation on the Java side makes REST calls to the present Python class to
    exchange action and state information.
    """

    @classmethod
    def get_argument_parser(
            cls,
    ) -> ArgumentParser:
        """
        Parse arguments.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        return parser

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

        parsed_args, unparsed_args = parse_arguments(cls, args)

        robocode = RobocodeEnvironment(
            random_state=random_state,
            **vars(parsed_args)
        )

        return robocode, unparsed_args

    def init_state_from_rest_put_dict(
            self,
            rest_request_dict: Dict[Any, Any]
    ) -> Tuple[MdpState, Reward]:
        """
        Initialize a state from the dictionary provided by the REST PUT (e.g., for setting and resetting the state).

        :param rest_request_dict: REST PUT dictionary.
        :return: 2-tuple of the state and reward.
        """

        # process all events that came through with the put
        dead = False
        won = False
        bullet_power_hit_others = 0
        bullet_power_hit_self = 0
        for event_wrapper in rest_request_dict['events']:
            event = event_wrapper['event']
            event_type = event_wrapper['type']
            if event_type == 'DeathEvent':
                dead = True
            elif event_type == 'WinEvent':
                won = True
            elif event_type == 'BulletHitEvent':
                bullet_power_hit_others += event['bullet']['power']
            elif event_type == 'HitByBulletEvent':
                bullet_power_hit_self += event['bullet']['power']

        # the round terminates either with death or victory -- it's a harsh world out there.
        terminal = dead or won

        # initialize the state
        state = RobocodeState(
            **rest_request_dict['state'],
            actions=self.robot_actions,
            terminal=terminal
        )

        # reward structure
        if dead:
            reward_value = -100.0
        elif won:
            reward_value = 100.0
        else:
            reward_value = bullet_power_hit_others - bullet_power_hit_self

        reward = Reward(
            i=None,
            r=reward_value
        )

        return state, reward

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            port: int,
            logging: bool
    ):
        """
        Initialize the MDP environment.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param port: Port to serve REST endpoints on.
        :param logging: Whether or not to print Flask logging messages to console.
        """

        super().__init__(
            name='robocode',
            random_state=random_state,
            T=T,
            port=port,
            logging=logging
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
            battle_field_height: float,
            battle_field_width: float,
            energy: float,
            gun_cooling_rate: float,
            gun_heading: float,
            gun_heat: float,
            heading: float,
            height: float,
            num_rounds: int,
            num_sentries: int,
            others: int,
            radar_heading: float,
            round_num: int,
            sentry_border_size: int,
            time: float,
            velocity: float,
            width: float,
            x: float,
            y: float,
            actions: List[Action],
            terminal: bool
    ):
        """
        Initialize the state.

        :param battle_field_height: Battle field height (pixels).
        :param battle_field_width: Battle field width (pixels).
        :param energy: Robot energy.
        :param gun_cooling_rate: Gun cooling rate (units per turn).
        :param gun_heading: Gun heading (degrees).
        :param gun_heat: Gun heat.
        :param heading: Robot heading (degrees).
        :param height: Robot height (pixels).
        :param num_rounds: Number of rounds in battle.
        :param num_sentries: Number of sentries left.
        :param others: Number of other robots left.
        :param radar_heading: Radar heading (pixels).
        :param round_num: Current round number.
        :param sentry_border_size: Sentry border size.
        :param time: Current turn of current round.
        :param velocity: Robot velocity (pixels per turn).
        :param width: Robot width (pixels).
        :param x: Robot x position.
        :param y: Robot y position.
        :param actions: List of actions that can be taken.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=None,
            AA=actions,
            terminal=terminal
        )

        self.battle_field_height = battle_field_height
        self.battle_field_width = battle_field_width
        self.energy = energy
        self.gun_cooling_rate = gun_cooling_rate
        self.gun_heading = gun_heading
        self.gun_heat = gun_heat
        self.heading = heading
        self.height = height
        self.num_rounds = num_rounds
        self.num_sentries = num_sentries
        self.others = others
        self.radar_heading = radar_heading
        self.round_num = round_num
        self.sentry_border_size = sentry_border_size
        self.time = time
        self.velocity = velocity
        self.width = width
        self.x = x
        self.y = y


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
