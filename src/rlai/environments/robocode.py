import math
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.agents.mdp import MdpAgent, StochasticMdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.environments.network import TcpMdpEnvironment
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments
from rlai.value_estimation.function_approximation.models.feature_extraction import FeatureExtractor


@rl_text(chapter='Rewards', page=1)
class RobocodeAimingReward(Reward):
    """
    Robocode aiming reward.
    """

    def __init__(
            self,
            i,
            r,
            bullet_id_fired_event,
            bullet_hit_events,
            bullet_missed_events
    ):
        """
        Initialize the reward.

        :param i: Identifier for the reward.
        :param r: Reward value.
        :param bullet_id_fired_event: Bullet identifier firing events.
        :param bullet_hit_events: Bullet hit events.
        :param bullet_missed_events: Bullet missed events.
        """

        super().__init__(
            i=i,
            r=r
        )

        self.bullet_id_fired_event = bullet_id_fired_event
        self.bullet_hit_events = bullet_hit_events
        self.bullet_missed_events = bullet_missed_events


@rl_text(chapter='Rewards', page=1)
class RobocodeMovementReward(Reward):
    """
    Robocode movement reward.
    """

    def __init__(
            self,
            i,
            r
    ):
        """
        Initialize the reward.

        :param i: Identifier for the reward.
        :param r: Reward value.
        """

        super().__init__(
            i=i,
            r=r
        )


@rl_text(chapter='Agents', page=1)
class RobocodeAgent(StochasticMdpAgent):
    """
    Robocode agent.
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
            random_state: RandomState,
            pi: Optional[Policy]
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param pi: Policy.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        agents = [
            RobocodeAgent(
                name=f'Robocode (gamma={parsed_args.gamma})',
                random_state=random_state,
                pi=pi,
                **vars(parsed_args)
            )
        ]

        return agents, unparsed_args

    def shape_reward(
            self,
            reward: Reward,
            first_t: int,
            final_t: int
    ) -> List[Tuple[int, float]]:
        """
        Shape a reward value that has been obtained. Reward shaping entails the calculation of time steps at which
        returns should be updated along with the weighted reward for each. This function applies exponential discounting
        based on the value of gamma specified in the current agent (i.e., the traditional reward shaping approach
        discussed by Sutton and Barto). Subclasses are free to override the current function and shape rewards as needed
        for the task at hand.

        The current function overrides the super-class implementation to add time shifting of rewards related to gun
        aiming.

        :param reward: Obtained reward.
        :param first_t: First time step at which to shape reward value.
        :param final_t: Final time step at which to shape reward value.
        :return: List of time steps for which returns should be updated, along with shaped rewards.
        """

        # if we received an aiming reward, then shift the reward backward in time to the evaluation time step of the
        # most recent bullet event (hit or missed).
        if isinstance(reward, RobocodeAimingReward) and len(reward.bullet_hit_events) + len(reward.bullet_missed_events) > 0:

            shifted_final_t = max([
                reward.bullet_id_fired_event[bullet_event['bullet']['bulletId']]['step']
                for bullet_event in reward.bullet_hit_events + reward.bullet_missed_events
            ])

            # also shift the value of first_t to maintain proper n-step update intervals
            shift_amount = shifted_final_t - final_t
            first_t = max(0, first_t + shift_amount)
            final_t = shifted_final_t

        return super().shape_reward(
            reward=reward,
            first_t=first_t,
            final_t=final_t
        )

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            pi: Policy,
            gamma: float
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param pi: Policy.
        :param gamma: Discount.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            pi=pi,
            gamma=gamma
        )


@rl_text(chapter='Environments', page=1)
class RobocodeEnvironment(TcpMdpEnvironment):
    """
    Robocode environment. The Java implementation of Robocode runs alongside the current environment, and a specialized
    robot implementation on the Java side makes TCP calls to the present Python class to exchange action and state
    information.
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

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> MdpState:
        """
        Reset the environment for a new run.

        :param agent: Agent.
        :return: Initial state.
        """

        initial_state = super().reset_for_new_run(agent)

        self.previous_state = None
        self.bullet_id_fired_event.clear()

        return initial_state

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

        # bullet power that hit self
        bullet_power_hit_self = sum([
            bullet_event['bullet']['power']
            for bullet_event in event_type_events.get('HitByBulletEvent', [])
        ])

        # sum up bullet power that hit others
        bullet_hit_events = event_type_events.get('BulletHitEvent', [])
        bullet_power_hit_others = sum([
            bullet_event['bullet']['power']
            for bullet_event in bullet_hit_events
        ])

        # sum up bullet power that missed others
        bullet_missed_events = event_type_events.get('BulletMissedEvent', [])
        bullet_power_missed = sum([
            bullet_event['bullet']['power']
            for bullet_event in bullet_missed_events
        ])

        # keep track of how much bullet power has missed the opponent since we last recorded a hit
        if self.previous_state is None:
            bullet_power_missed_since_previous_hit = 0.0
        elif bullet_power_hit_others > 0.0:
            bullet_power_missed_since_previous_hit = 0.0
        else:
            bullet_power_missed_since_previous_hit = self.previous_state.bullet_power_missed_since_previous_hit + bullet_power_missed

        # the round terminates either with death or victory -- it's a harsh world out there.
        dead = len(event_type_events.get('DeathEvent', [])) > 0
        won = len(event_type_events.get('WinEvent', [])) > 0
        terminal = dead or won

        # initialize the state
        state = RobocodeState(
            **client_dict['state'],
            bullet_power_hit_self=bullet_power_hit_self,
            bullet_power_hit_self_cumulative=bullet_power_hit_self + (0.0 if self.previous_state is None else self.previous_state.bullet_power_hit_self * 0.99),
            bullet_power_hit_others=bullet_power_hit_others,
            bullet_power_missed=bullet_power_missed,
            bullet_power_missed_since_previous_hit=bullet_power_missed_since_previous_hit,
            events=event_type_events,
            AA=self.robot_actions,
            terminal=terminal
        )

        # movement reward
        reward = RobocodeMovementReward(
            i=None,
            r=1.0 if bullet_power_hit_self == 0.0 else -10.0
        )

        # # aiming reward
        # # hit others and don't miss
        # reward_value = bullet_power_hit_others - bullet_power_missed
        #
        # # store bullet firing events so that we can pull out information related to them at a later time step (e.g.,
        # # when they hit or miss). add the evaluation time step to each event. the bullet events have times associated
        # # with them that are provided by the robocode engine, but those are robocode turns and there isn't always a
        # # perfect 1:1 between robocode turns and evaluation time steps.
        # self.bullet_id_fired_event.update({
        #     bullet_fired_event['bullet']['bulletId']: {
        #         **bullet_fired_event,
        #         'step': t
        #     }
        #     for bullet_fired_event in event_type_events.get('BulletFiredEvent', [])
        # })
        #
        # reward = RobocodeAimingReward(
        #     i=None,
        #     r=reward_value,
        #     bullet_id_fired_event=self.bullet_id_fired_event,
        #     bullet_hit_events=bullet_hit_events,
        #     bullet_missed_events=bullet_missed_events
        # )

        self.previous_state = state

        return state, reward

        # old/obsolete reward signals

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

        # use the following actions for aiming training
        # action_name_action_value_list = [
        #     ('turnRadarLeft', 5.0),
        #     ('turnRadarRight', 5.0),
        #     ('turnGunLeft', 5.0),
        #     ('turnGunRight', 5.0),
        #     ('fire', 1.0)
        # ]

        # use the following actions for movement training
        action_name_action_value_list = [
            ('ahead', 25.0),
            ('back', 25.0),
            ('turnLeft', 10.0),
            ('turnRight', 10.0),
            ('turnRadarLeft', 5.0),
            ('turnRadarRight', 5.0)
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
            bullet_power_hit_self: float,
            bullet_power_hit_self_cumulative: float,
            bullet_power_hit_others: float,
            bullet_power_missed: float,
            bullet_power_missed_since_previous_hit: float,
            events: Dict[str, List[Dict]],
            AA: List[Action],
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
        :param bullet_power_hit_self: Bullet power that hit self.
        :param bullet_power_hit_self_cumulative: Cumulative bullet power that hit self, including a discounted sum of prior values.
        :param bullet_power_hit_others: Bullet power that hit others.
        :param bullet_power_missed: Bullet power that missed.
        :param bullet_power_missed_since_previous_hit: Bullet power that has missed since previous hit.
        :param events: List of events sent to the robot since the previous time the state was set.
        :param AA: List of actions that can be taken.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=None,
            AA=AA,
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
        self.bullet_power_hit_self = bullet_power_hit_self
        self.bullet_power_hit_self_cumulative = bullet_power_hit_self_cumulative
        self.bullet_power_hit_others = bullet_power_hit_others
        self.bullet_power_missed = bullet_power_missed
        self.bullet_power_missed_since_previous_hit = bullet_power_missed_since_previous_hit
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

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset the feature extractor for a new run.

        :param state: Initial state.
        """

        super().reset_for_new_run(state)

        self.most_recent_scanned_robot = None

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

        self.set_most_recent_scanned_robot(states)

        if self.most_recent_scanned_robot is None:
            enemy_bearing_from_self = None
            enemy_distance_from_self = None
        else:
            # the bearing comes to us as [-180,180], whichever is closest. normalize to [0,360]. this bearing is
            # relative to our heading, so degrees from north to thee enemy would be our heading plus this value.
            enemy_bearing_from_self = self.normalize(math.degrees(self.most_recent_scanned_robot['bearing']))
            enemy_distance_from_self = self.most_recent_scanned_robot['distance']

        X = np.array([

            [
                feature_value

                # extract each possible action for the current state. only the action that is paired with the current
                # state will have nonzero values. this is essentially forming one-hot-action interaction terms between
                # the state features and the categorical actions.
                for action_to_extract in state.AA

                # extract feature values
                for feature_value in self.get_feature_values(state, action, action_to_extract, enemy_bearing_from_self, enemy_distance_from_self)
            ]

            # the state and actions come to us in pairs
            for state, action in zip(states, actions)
        ])

        return X

    # def get_feature_action_names(
    #         self
    # ) -> Tuple[List[str], List[str]]:
    #     """
    #     Get names of extracted features and actions.
    #
    #     :return: 2-tuple of (1) list of feature names and (2) list of action names.
    #     """
    #
    #     return (
    #         ['action_intercept', 'has_enemy_bearing', 'aim_lock'],
    #         [a.name for a in self.actions]
    #     )

    def set_most_recent_scanned_robot(
            self,
            states: List[RobocodeState]
    ):
        """
        Set the most recent scanned robot from a list of states.

        :param states: States.
        """

        self.most_recent_scanned_robot: Dict = next((

            # take the final element of the list, which should be the most recent event.
            state.events['ScannedRobotEvent'][-1]

            # reverse in case multiple states are given
            for state in reversed(states)

            # events might not contain scanned robot events
            if 'ScannedRobotEvent' in state.events

        ), None)

    def get_feature_values(
            self,
            state: RobocodeState,
            action: Action,
            action_to_extract: Action,
            enemy_bearing_from_self: Optional[float],
            enemy_distance_from_self: Optional[float]
    ) -> List[float]:
        """
        Get feature values for a state-action pair, coded one-hot for a particular action to extract.

        :param state: State.
        :param action: Action.
        :param action_to_extract: Action to one-hot encode.
        :param enemy_bearing_from_self: Bearing of enemy from self, or None if no enemy has been scanned.
        :param enemy_distance_from_self: Distance of enemy from self, or None if no enemy has been scanned.
        :return: List of floating-point feature values.
        """

        if action_to_extract.name == 'ahead' or action_to_extract.name == 'back':

            if action == action_to_extract:
                feature_values = [

                    # intercept
                    1.0,

                    # indicator (0/1):  we have a bearing on the enemy
                    0.0 if enemy_bearing_from_self is None else 1.0,

                    # discounted cumulative bullet power
                    state.bullet_power_hit_self_cumulative,

                    # hit a wall with front/back of robot
                    1.0 if any(-90 < e['bearing'] < 90 for e in state.events.get('HitWallEvent', [])) else 0.0,
                    1.0 if any(-90 > e['bearing'] > 90 for e in state.events.get('HitWallEvent', [])) else 0.0,

                    # hit a robot with front/back of robot
                    1.0 if any(-90 < e['bearing'] < 90 for e in state.events.get('HitRobotEvent', [])) else 0.0,
                    1.0 if any(-90 > e['bearing'] > 90 for e in state.events.get('HitRobotEvent', [])) else 0.0,

                    0.0 if enemy_bearing_from_self is None else
                    self.funnel(
                        self.get_shortest_degree_change(
                            state.heading,
                            self.normalize(state.heading + enemy_bearing_from_self + 90.0)
                        ),
                        True,
                        15.0
                    ),

                    0.0 if enemy_bearing_from_self is None else
                    self.funnel(
                        self.get_shortest_degree_change(
                            state.heading,
                            self.normalize(state.heading + enemy_bearing_from_self - 90.0)
                        ),
                        True,
                        15.0
                    )

                ]
            else:
                feature_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        elif action_to_extract.name == 'turnLeft' or action_to_extract.name == 'turnRight':

            if action == action_to_extract:
                feature_values = [

                    # intercept
                    1.0,

                    # indicator (0/1):  we have a bearing on the enemy
                    0.0 if enemy_bearing_from_self is None else 1.0,

                    0.0 if enemy_bearing_from_self is None else
                    self.sigmoid(
                        self.get_shortest_degree_change(
                            state.heading,
                            self.normalize(state.heading + enemy_bearing_from_self + 90.0)
                        ),
                        15.0
                    ),

                    0.0 if enemy_bearing_from_self is None else
                    self.sigmoid(
                        self.get_shortest_degree_change(
                            state.heading,
                            self.normalize(state.heading + enemy_bearing_from_self - 90.0)
                        ),
                        15.0
                    )

                ]
            else:
                feature_values = [0.0, 0.0, 0.0, 0.0]

        elif action_to_extract.name.startswith('turnRadar'):

            if action == action_to_extract:
                feature_values = [

                    # intercept
                    1.0,

                    # indicator (0/1):  we have a bearing on the enemy
                    0.0 if enemy_bearing_from_self is None else 1.0,

                    # squash lateral distance into [-1.0, 1.0]
                    0.0 if enemy_bearing_from_self is None else
                    self.sigmoid(
                        self.get_lateral_distance(
                            -self.get_shortest_degree_change(
                                state.radar_heading,
                                self.normalize(state.heading + enemy_bearing_from_self)
                            ),
                            enemy_distance_from_self
                        ),
                        20.0
                    )

                ]
            else:
                feature_values = [0.0, 0.0, 0.0]

        elif action_to_extract.name.startswith('turnGun'):

            if action == action_to_extract:
                feature_values = [

                    # intercept
                    1.0,

                    # indicator (0/1):  we have a bearing on the enemy
                    0.0 if enemy_bearing_from_self is None else 1.0,

                    # squash lateral distance into [-1.0, 1.0]
                    0.0 if enemy_bearing_from_self is None else
                    self.sigmoid(
                        self.get_lateral_distance(
                            -self.get_shortest_degree_change(
                                state.gun_heading,
                                self.normalize(state.heading + enemy_bearing_from_self)
                            ),
                            enemy_distance_from_self
                        ),
                        20.0
                    )
                ]
            else:
                feature_values = [0.0, 0.0, 0.0]

        elif action_to_extract.name == 'fire':

            if action == action_to_extract:
                feature_values = [

                    # intercept
                    1.0,

                    # indicator (0/1):  we have a bearing on the enemy
                    0.0 if enemy_bearing_from_self is None else 1.0,

                    # funnel lateral distance to 0.0
                    0.0 if enemy_bearing_from_self is None else
                    self.funnel(
                        self.get_lateral_distance(
                            -self.get_shortest_degree_change(
                                state.gun_heading,
                                self.normalize(state.heading + enemy_bearing_from_self)
                            ),
                            enemy_distance_from_self
                        ),
                        True,
                        20.0
                    )
                ]
            else:
                feature_values = [0.0, 0.0, 0.0]

        else:  # pragma no cover
            raise ValueError(f'Unknown action:  {action}')

        return feature_values

    @classmethod
    def is_clockwise_move(
            cls,
            start_heading: float,
            end_heading: float
    ) -> bool:
        """
        Check whether moving from one heading to another would be a clockwise movement.

        :param start_heading: Start heading (degrees [0, 360]).
        :param end_heading: End heading (degrees [0, 360]).
        :return: True if moving from `start_heading` to `end_heading` would move in the clockwise direction.
        """

        return cls.get_shortest_degree_change(start_heading, end_heading) > 0.0

    @staticmethod
    def get_shortest_degree_change(
            start_heading: float,
            end_heading: float
    ) -> float:
        """
        Get the shortest degree change to go from one heading to another.

        :param start_heading: Start heading (degrees [0, 360]).
        :param end_heading: End heading (degrees [0, 360]).
        :return: Shortest degree change to move from `start_heading` to `end_heading` (degrees, [-180, 180]).
        """

        # clockwise change is always positive, and counterclockwise change is always negative.

        if end_heading > start_heading:
            clockwise_change = end_heading - start_heading
            counterclockwise_change = clockwise_change - 360.0
        elif start_heading > end_heading:
            counterclockwise_change = end_heading - start_heading
            clockwise_change = counterclockwise_change + 360.0
        else:
            clockwise_change = 0.0
            counterclockwise_change = 0.0

        if abs(clockwise_change) < abs(counterclockwise_change):
            return clockwise_change
        else:
            return counterclockwise_change

    @staticmethod
    def funnel(
            x: float,
            up: bool,
            scale: float
    ) -> float:
        """
        Impose funnel function on a value.

        :param x: Value.
        :param up: Whether to funnel up (True) or down (False).
        :param scale: Scale.
        :return: Funnel value in [-1.0, 1.0].
        """

        v = 4.0 * ((1.0 / (1.0 + np.exp(-abs(x / scale)))) - 0.75)

        if up:
            v = -v

        return v

    @staticmethod
    def sigmoid(
            x: float,
            scale: float
    ) -> float:
        """
        Impose sigmoid function on a value.

        :param x: Value.
        :param scale: Scale.
        :return: Sigmoid value in [-1.0, 1.0].
        """

        return 2.0 * ((1.0 / (1.0 + np.exp(-(x / scale)))) - 0.5)

    @staticmethod
    def get_lateral_distance(
            offset_degrees: float,
            distance: float
    ) -> float:
        """
        Get lateral distance to a target, at an offset degree and distance.

        :param offset_degrees: Offset degree.
        :param distance: Distance.
        :return: Lateral distance
        """

        if offset_degrees <= -90:
            lateral_distance = -np.inf
        elif offset_degrees >= 90:
            lateral_distance = np.inf
        else:
            lateral_distance = math.tan(math.radians(offset_degrees)) * distance

        return lateral_distance

    @staticmethod
    def normalize(
            degrees: float
    ) -> float:
        """
        Normalizee degress to be [0, 360].

        :param degrees: Degrees.
        :return: Normalized degrees.
        """

        degrees = degrees % 360.0

        if degrees < 0 or degrees > 360:
            raise ValueError(f'Failed to normalize degrees:  {degrees}')

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
