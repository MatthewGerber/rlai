import math
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.environments.network import TcpMdpEnvironment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments
from rlai.value_estimation.function_approximation.models.feature_extraction import (
    NonstationaryFeatureScaler,
    FeatureExtractor
)


@rl_text(chapter='Environments', page=1)
class RobocodeEnvironment(TcpMdpEnvironment):
    """
    Robocode environment. The official Java implementation of Robocode runs alongside the current environment, and a
    specialized robot implementation on the Java side makes TCP calls to the present Python class to exchange action and
    state information.
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

    def extract_state_and_reward_from_client_dict(
            self,
            client_dict: Dict[Any, Any],
            t: int
    ) -> Tuple[MdpState, Reward]:
        """
        Extract the state and reward from a client dict.

        :param client_dict: Client dictionary.
        :param t: Current time step.
        :return: 2-tuple of the state and reward.
        """

        event_type_events: Dict[str, List[Dict]] = client_dict['events']

        # the round terminates either with death or victory -- it's a harsh world out there.
        dead = len(event_type_events.get('DeathEvent', [])) > 0
        won = len(event_type_events.get('WinEvent', [])) > 0
        terminal = dead or won

        # initialize the state
        state = RobocodeState(
            **client_dict['state'],
            events=event_type_events,
            actions=self.robot_actions,
            terminal=terminal
        )

        # hang on to bullet firing events. add rlai step to each event.
        self.bullet_id_fired_event.update({
            bullet_fired_event['bullet']['bulletId']: {
                **bullet_fired_event,
                'step': t
            }
            for bullet_fired_event in event_type_events.get('BulletFiredEvent', [])
        })

        # reward structure

        # ... don't get hit
        # bullet_power_hit_self = sum([
        #     bullet_event['bullet']['power']
        #     for bullet_event in event_type_events.get('HitByBulletEvent', [])
        # ])

        # ... death seems bad, and victories good.
        # if dead:
        #     reward_value = -100.0
        # elif won:
        #     reward_value = 100.0
        # else:

        # ... probably a bad idea here, since energy loss can be recovered upon bullet impact, which is good.
        # if self.previous_state is None:
        #     reward_value = 0.0
        # else:
        #     reward_value = state.energy - self.previous_state.energy

        # ... hit others
        bullet_hit_events = event_type_events.get('BulletHitEvent', [])
        bullet_power_hit_others = sum([
            bullet_event['bullet']['power']
            for bullet_event in bullet_hit_events
        ])

        # ... don't miss
        bullet_missed_events = event_type_events.get('BulletMissedEvent', [])
        bullet_power_missed = sum([
            bullet_event['bullet']['power']
            for bullet_event in bullet_missed_events
        ])

        reward_value = bullet_power_hit_others - bullet_power_missed

        # delay to most recent bullet event of either kind (hit or missed)
        if len(bullet_hit_events) + len(bullet_missed_events) == 0:
            reward_shift_steps = 0
        else:
            reward_shift_steps = max([
                self.bullet_id_fired_event[bullet_event['bullet']['bulletId']]['step']
                for bullet_event in bullet_hit_events + bullet_missed_events
            ]) - t

        # clear bullet firing event for any bullet that hit another robot, missed, or hit another bullet.
        [
            self.bullet_id_fired_event.pop(bullet_event['bullet']['bulletId'])
            for bullet_event in bullet_hit_events + bullet_missed_events + event_type_events.get('BulletHitBulletEvent', [])
        ]

        reward = Reward(
            i=None,
            r=reward_value,
            shift_steps=reward_shift_steps
        )

        self.previous_state = state

        return state, reward

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            port: int
    ):
        """
        Initialize the Robocode environment.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param port: Port to serve REST endpoints on.
        """

        super().__init__(
            name='robocode',
            random_state=random_state,
            T=T,
            port=port
        )

        action_name_action_value_list = [
            # ('ahead', 25.0),
            # ('back', 25.0),
            # ('turnLeft', 5.0),
            # ('turnRight', 5.0),
            ('turnRadarLeft', 5.0),
            # ('turnRadarRight', 5.0),
            ('turnGunLeft', 5.0),
            ('turnGunRight', 5.0),
            ('fire', 1.0)
        ]

        self.robot_actions = [
            RobocodeAction(i, action_name, action_value)
            for i, (action_name, action_value) in enumerate(action_name_action_value_list)
        ]

        self.previous_state: Optional[RobocodeState] = None
        self.bullet_id_fired_event: Dict[str, Dict] = {}


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
            events: Dict[str, List[Dict]],
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
        :param events: List of events sent to the robot since the previous time the state was set.
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
        self.events = events


@rl_text(chapter='Feature Extractors', page=1)
class RobocodeFeatureExtractor(FeatureExtractor):
    """
    Robocode feature extractor.
    """

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

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: RobocodeEnvironment
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        fex = RobocodeFeatureExtractor(
            environment=environment
        )

        return fex, unparsed_args

    def extract(
            self,
            states: List[RobocodeState],
            actions: List[Action],
            for_fitting: bool
    ) -> np.ndarray:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param for_fitting: Whether the extracted features will be used for fitting (True) or prediction (False).
        :return: State-feature numpy.ndarray.
        """

        self.check_state_and_action_lists(states, actions)

        # update the most recent scanned robot event, which is our best estimate as to where the enemy robot is
        # located.
        self.most_recent_scanned_robot: Dict = next((
            state.events['ScannedRobotEvent'][0]
            for state in states
            if 'ScannedRobotEvent' in state.events
        ), self.most_recent_scanned_robot)

        # bearing is relative to our heading
        if self.most_recent_scanned_robot is None:
            bearing_from_self = None
        else:
            bearing_from_self = math.degrees(self.most_recent_scanned_robot['bearing'])

        X = np.array([

            [
                feature_value
                for action_to_extract in self.actions
                for feature_value in self.get_feature_values(state, action, action_to_extract, bearing_from_self)
            ]

            for state, action in zip(states, actions)
        ])

        X = self.feature_scaler.scale_features(X, for_fitting)

        return X

    def get_feature_values(
            self,
            state: RobocodeState,
            action: Action,
            action_to_extract: Action,
            bearing_from_self: Optional[float]
    ) -> List[float]:
        """
        Get feature values for a state-action pair, coded one-hot for a particular action to extract.

        :param state: State.
        :param action: Action.
        :param action_to_extract: Action to one-hot.
        :param bearing_from_self: Bearing of enemy from self, or None if no enemy has been scanned yet.
        :return: List of floating-point feature values.
        """

        if action_to_extract.name.startswith('turnRadar'):

            if action == action_to_extract:
                feature_values = [

                    # provide a constant, so that each action has its own intercept term.
                    1.0,

                    # indicator as to whether or not we have a bearing on the enemy
                    0.0 if bearing_from_self is None else 1.0

                ]
            else:
                feature_values = [0.0, 0.0]

        elif action_to_extract.name.startswith('turnGun'):

            if action == action_to_extract:
                feature_values = [

                    # provide a constant, so that each action has its own intercept term.
                    1.0,

                    # indicator as to whether or not we have a bearing on the enemy
                    0.0 if bearing_from_self is None else 1.0,

                    # indicator variable that is 1 if the gun's current heading (w.r.t to us) is counterclockwise from the
                    # enemy robot's bearing (w.r.t. us). zero if no enemy has been scanned.
                    0.0 if bearing_from_self is None else int(state.gun_heading - self.normalize(state.heading + bearing_from_self) < 0)

                ]
            else:
                feature_values = [0.0, 0.0, 0.0]

        elif action_to_extract.name == 'fire':

            if action == action_to_extract:
                feature_values = [

                    # provide a constant, so that each action has its own intercept term.
                    1.0,

                    # indicator as to whether or not we have a bearing on the enemy
                    0.0 if bearing_from_self is None else 1.0,

                    # sqrt(abs) the difference as a measure of how close the gun is to pointing at the enemy. zero if no
                    # enemy has been scanned.
                    0.0 if bearing_from_self is None else math.sqrt(abs(state.gun_heading - self.normalize(state.heading + bearing_from_self)))

                ]
            else:
                feature_values = [0.0, 0.0, 0.0]
        else:
            raise ValueError(f'Unknown action:  {action.name}')

        return feature_values

    @staticmethod
    def normalize(
            degrees: float
    ) -> float:

        if degrees > 360:
            degrees -= 360
        elif degrees < 0:
            degrees += 360

        return degrees

    def __init__(
            self,
            environment: RobocodeEnvironment
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        super().__init__(
            environment=environment
        )

        self.actions = environment.robot_actions
        self.most_recent_scanned_robot = None
        self.feature_scaler = NonstationaryFeatureScaler(
            num_observations_refit_feature_scaler=2000,
            refit_history_length=100000,
            refit_weight_decay=0.99999
        )


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
