import math
from argparse import ArgumentParser
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from PyQt6.QtWidgets import QApplication
from numpy.random import RandomState

from rlai.core import Reward, Action, ContinuousMultiDimensionalAction, MdpState, MdpAgent, Environment
from rlai.core.environments.mdp import ContinuousMdpEnvironment
from rlai.core.environments.network import TcpMdpEnvironment
from rlai.docs import rl_text
from rlai.models.feature_extraction import FeatureExtractor, StationaryFeatureScaler
from rlai.policy_gradient import ParameterizedMdpAgent
from rlai.policy_gradient.policies import ParameterizedPolicy
from rlai.state_value import StateValueEstimator
from rlai.state_value.function_approximation.models.feature_extraction import StateFeatureExtractor
from rlai.utils import parse_arguments


@rl_text(chapter='Rewards', page=1)
class RobocodeReward(Reward):
    """
    Robocode reward.
    """

    def __init__(
            self,
            i: Optional[int],
            r: float,
            gun_reward: float,
            movement_reward: float,
            bullet_id_fired_event: Dict[str, Dict],
            bullet_hit_events: List[Dict],
            bullet_missed_events: List[Dict]
    ):
        """
        Initialize the reward.

        :param i: Identifier for the reward.
        :param r: Total reward value.
        :param gun_reward: Portion of total reward due to aiming and firing the gun (e.g., hitting the enemy).
        :param movement_reward: Portion of total reward due to movement (e.g., avoiding being hit).
        :param bullet_id_fired_event: Bullet identifier firing events.
        :param bullet_hit_events: Bullet hit events.
        :param bullet_missed_events: Bullet missed events.
        """

        super().__init__(
            i=i,
            r=r
        )

        self.gun_reward = gun_reward
        self.movement_reward = movement_reward
        self.bullet_id_fired_event = bullet_id_fired_event
        self.bullet_hit_events = bullet_hit_events
        self.bullet_missed_events = bullet_missed_events


@rl_text(chapter='Agents', page=1)
class RobocodeAgent(ParameterizedMdpAgent):
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

    def shape_reward(
            self,
            reward: Reward,
            first_t: int,
            final_t: int
    ) -> Dict[int, float]:
        """
        Shape a reward value that has been obtained. Reward shaping entails the calculation of time steps at which
        returns should be updated along with the weighted reward for each. The super-class implementation applies
        exponential discounting based on the value of gamma specified in the current agent (i.e., the traditional reward
        shaping approach discussed by Sutton and Barto). The current function overrides the super-class implementation
        to add time shifting of rewards related to gun aiming.

        :param reward: Obtained reward.
        :param first_t: First time step at which to shape reward value.
        :param final_t: Final time step at which to shape reward value.
        :return: Dictionary of time steps and associated shaped rewards.
        """

        # the typical case is to shape a reward returned by the robocode environment
        if isinstance(reward, RobocodeReward):

            # the reward will always have a movement component. shape it in the standard way.
            t_shaped_reward = super().shape_reward(
                reward=Reward(None, reward.movement_reward),
                first_t=first_t,
                final_t=final_t
            )

            # if the reward has a gun component, then shift the gun component backward in time based on when the
            # bullet(s) were fired.
            if len(reward.bullet_hit_events) + len(reward.bullet_missed_events) > 0:

                # get the most recent origination time step for bullets associated with the given reward. this is the
                # time step at which the bullet was fired. we're going to shape the reward backward in time starting
                # here.
                shifted_final_t = max([
                    reward.bullet_id_fired_event[bullet_event['bullet']['bulletId']]['step']
                    for bullet_event in reward.bullet_hit_events + reward.bullet_missed_events
                ])

                # also shift the value of first_t to maintain proper n-step update intervals
                shift_amount = shifted_final_t - final_t
                first_t = max(0, first_t + shift_amount)
                final_t = shifted_final_t

                t_shaped_reward.update({
                    t: t_shaped_reward.get(t, 0.0) + shaped_reward
                    for t, shaped_reward in super().shape_reward(
                        reward=Reward(None, reward.gun_reward),
                        first_t=first_t,
                        final_t=final_t
                    ).items()
                })

        # the following case is not currently used.
        elif isinstance(reward, Reward):  # pragma no cover

            t_shaped_reward = super().shape_reward(
                reward=reward,
                first_t=first_t,
                final_t=final_t
            )

        # a standard reward is returned by the underlying networked environment if the game client disconnects.
        else:  # pragma no cover
            t_shaped_reward = {}

        return t_shaped_reward

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            pi: ParameterizedPolicy,
            gamma: float,
            v_S: StateValueEstimator
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param pi: Policy.
        :param gamma: Discount.
        :param v_S: Baseline state-value estimator.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            pi=pi,
            gamma=gamma,
            v_S=v_S
        )


@rl_text(chapter='Environments', page=1)
class RobocodeEnvironment(TcpMdpEnvironment, ContinuousMdpEnvironment):
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

        parser.add_argument(
            '--bullet-power-decay',
            type=float,
            help='Exponential decay rate for cumulative bullet power values.'
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

        robocode = cls(
            random_state=random_state,
            **vars(parsed_args)
        )

        return robocode, unparsed_args

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            port: int,
            bullet_power_decay: float
    ):
        """
        Initialize the Robocode environment.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param port: Port to serve networked environment on.
        :param bullet_power_decay: Exponential decay rate for cumulative bullet power values.
        """

        super().__init__(
            name='robocode',
            random_state=random_state,
            T=T,
            port=port
        )

        self.bullet_power_decay = bullet_power_decay

        min_values = np.array([
            -100.0,  # RobocodeAction.AHEAD
            -180.0,  # RobocodeAction.TURN_LEFT
            -180.0,  # RobocodeAction.TURN_RADAR_LEFT
            -180.0,  # RobocodeAction.TURN_GUN_LEFT
            0.0  # RobocodeAction.FIRE
        ])

        max_values = np.array([
            100.0,  # RobocodeAction.AHEAD
            180.0,  # RobocodeAction.TURN_LEFT
            180.0,  # RobocodeAction.TURN_RADAR_LEFT
            180.0,  # RobocodeAction.TURN_GUN_LEFT
            5.0  # RobocodeAction.FIRE
        ])

        self.robot_actions: List[Action] = [
            ContinuousMultiDimensionalAction(
                value=None,
                min_values=min_values,
                max_values=max_values,
                name='Robocode continuous action'
            )
        ]

        self.previous_state: Optional[RobocodeState] = None
        self.bullet_id_fired_event: Dict[str, Dict] = {}

    def get_state_space_dimensionality(self) -> int:
        """
        Get state-space dimensionality.

        :return: Dimensionality.
        """

        return 15

    def get_state_dimension_names(self) -> List[str]:
        """
        Get state dimension names.

        :return: Names.
        """

        return [str(i) for i in range(self.get_state_space_dimensionality())]

    def get_action_space_dimensionality(self) -> int:
        """
        Get action-space dimensionality.

        :return: Dimensionality.
        """

        return 5

    def get_action_dimension_names(self) -> List[str]:
        """
        Get action dimension names.

        :return: Names.
        """

        return [str(i) for i in range(self.get_action_space_dimensionality())]

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> MdpState:
        """
        Reset the environment for a new run.

        :param agent: Agent.
        :return: Initial state.
        """

        # the call to super().reset_for_new_run will eventually call extract_state_and_reward_from_client_dict, and this
        # function depends on the following variables. reset them now, before the call happens, to ensure that the
        # variables reflect the reset at the time they're used.
        self.previous_state = None
        self.bullet_id_fired_event.clear()

        return super().reset_for_new_run(agent)

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

        QApplication.processEvents()

        # pull out robocode game events
        event_type_events: Dict[str, List[Dict]] = client_dict['events']

        # calculate number of turns that have passed since previous call to the current function. will be greater than
        # 1 any time the previous action took more than 1 turn to complete (e.g., moving long distances).
        if self.previous_state is None:
            turns_passed = 0
        else:
            turns_passed = client_dict['state']['time'] - self.previous_state.time

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
        bullet_power_missed_others = sum([
            bullet_event['bullet']['power']
            for bullet_event in bullet_missed_events
        ])

        # keep track of how much bullet power has missed the opponent since we last recorded a hit
        if self.previous_state is None:
            bullet_power_missed_others_since_previous_hit = 0.0
        elif bullet_power_hit_others > 0.0:
            bullet_power_missed_others_since_previous_hit = 0.0
        else:
            bullet_power_missed_others_since_previous_hit = self.previous_state.bullet_power_missed_others_since_previous_hit + bullet_power_missed_others

        # cumulative bullet power that has hit self, decaying over time.
        bullet_power_hit_self_cumulative = bullet_power_hit_self
        if self.previous_state is not None:
            bullet_power_hit_self_cumulative += self.previous_state.bullet_power_hit_self_cumulative * (self.bullet_power_decay ** turns_passed)

        # cumulative bullet power that has hit others, decaying over time.
        bullet_power_hit_others_cumulative = bullet_power_hit_others
        if self.previous_state is not None:
            bullet_power_hit_others_cumulative += self.previous_state.bullet_power_hit_others_cumulative * (self.bullet_power_decay ** turns_passed)

        # cumulative bullet power that has missed others, decaying over time.
        bullet_power_missed_others_cumulative = bullet_power_missed_others
        if self.previous_state is not None:
            bullet_power_missed_others_cumulative += self.previous_state.bullet_power_missed_others_cumulative * (self.bullet_power_decay ** turns_passed)

        # get most recent prior state that was at a different location than the current state. if there is no previous
        # state, then there is no such state.
        if self.previous_state is None:
            prior_state_different_location = None
        # if the previous state's location differs from the current location, then the previous state is what we want.
        elif self.previous_state.x != client_dict['state']['x'] or self.previous_state.y != client_dict['state']['y']:
            prior_state_different_location = self.previous_state
        # otherwise (if the previous and current state have the same location), then use the previous state's prior
        # state. this will be the case when we did something other than move on our previous turn.
        else:
            prior_state_different_location = self.previous_state.prior_state_different_location

        # most recent scanned enemy and how many turns ago it was scanned
        most_recent_scanned_robot = event_type_events.get('ScannedRobotEvent', [None])[-1]
        most_recent_scanned_robot_age_turns = None
        if most_recent_scanned_robot is None:
            if self.previous_state is not None and self.previous_state.most_recent_scanned_robot is not None:
                most_recent_scanned_robot = self.previous_state.most_recent_scanned_robot
                assert self.previous_state.most_recent_scanned_robot_age_turns is not None
                most_recent_scanned_robot_age_turns = self.previous_state.most_recent_scanned_robot_age_turns + turns_passed
        else:
            most_recent_scanned_robot_age_turns = 0

        # the round terminates either with death or victory
        dead = len(event_type_events.get('DeathEvent', [])) > 0
        won = len(event_type_events.get('WinEvent', [])) > 0
        terminal = dead or won

        state = RobocodeState(
            **client_dict['state'],
            bullet_power_hit_self=bullet_power_hit_self,
            bullet_power_hit_self_cumulative=bullet_power_hit_self_cumulative,
            bullet_power_hit_others=bullet_power_hit_others,
            bullet_power_hit_others_cumulative=bullet_power_hit_others_cumulative,
            bullet_power_missed_others=bullet_power_missed_others,
            bullet_power_missed_others_cumulative=bullet_power_missed_others_cumulative,
            bullet_power_missed_others_since_previous_hit=bullet_power_missed_others_since_previous_hit,
            events=event_type_events,
            previous_state=self.previous_state,
            prior_state_different_location=prior_state_different_location,
            most_recent_scanned_robot=most_recent_scanned_robot,
            most_recent_scanned_robot_age_turns=most_recent_scanned_robot_age_turns,
            AA=self.robot_actions,
            terminal=terminal,
            truncated=False
        )

        # calculate reward

        # store bullet firing events so that we can pull out information related to them at a later time step (e.g.,
        # when they hit or miss). add the rlai time step (t) to each event. there is a 1:1 between rlai time steps and
        # actions, but an action can extend over many robocode turns (e.g., movement). the bullet events have times
        # associated with them that correspond to robocode turns.
        self.bullet_id_fired_event.update({
            bullet_fired_event['bullet']['bulletId']: {
                **bullet_fired_event,
                'step': t
            }
            for bullet_fired_event in event_type_events.get('BulletFiredEvent', [])
        })

        # gun_reward = bullet_power_hit_others - bullet_power_missed_others
        # movement_reward = 1.0 if bullet_power_hit_self == 0.0 else -bullet_power_hit_self
        # total_reward = gun_reward + movement_reward
        # reward = RobocodeReward(
        #     i=None,
        #     r=total_reward,
        #     gun_reward=gun_reward,
        #     movement_reward=movement_reward,
        #     bullet_id_fired_event=self.bullet_id_fired_event,
        #     bullet_hit_events=bullet_hit_events,
        #     bullet_missed_events=bullet_missed_events
        # )

        # energy change reward...bullet firing will be penalized.
        reward = Reward(
            None,
            r=0.0 if self.previous_state is None else state.energy - self.previous_state.energy
        )

        # win/loss reward
        # reward = Reward(
        #     None,
        #     r=1.0 if won else -1.0 if dead else 0.0
        # )

        # hang on to the new state as the previous state, for the next call to extract.
        self.previous_state = state

        return state, reward


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
            bullet_power_hit_others_cumulative: float,
            bullet_power_missed_others: float,
            bullet_power_missed_others_cumulative: float,
            bullet_power_missed_others_since_previous_hit: float,
            events: Dict[str, List[Dict]],
            previous_state: Optional['RobocodeState'],
            prior_state_different_location: Optional['RobocodeState'],
            most_recent_scanned_robot: Optional[Dict],
            most_recent_scanned_robot_age_turns: Optional[int],
            AA: List[Action],
            terminal: bool,
            truncated: bool
    ):
        """
        Initialize the state.

        :param battle_field_height: Battlefield height (pixels).
        :param battle_field_width: Battlefield width (pixels).
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
        :param bullet_power_hit_self_cumulative: Cumulative bullet power that hit self, including a discounted sum of
        prior values.
        :param bullet_power_hit_others: Bullet power that hit others.
        :param bullet_power_hit_others_cumulative: Cumulative bullet power that hit others, including a discounted sum
        of prior values.
        :param bullet_power_missed_others: Bullet power that missed.
        :param bullet_power_missed_others_cumulative: Cumulative bullet power that missed, including a discounted sum of
        prior values.
        :param bullet_power_missed_others_since_previous_hit: Bullet power that has missed since previous hit.
        :param events: List of events sent to the robot since the previous time the state was set.
        :param previous_state: Previous state.
        :param prior_state_different_location: Most recent prior state at a different location than the current state.
        :param most_recent_scanned_robot: Scanned enemy.
        :param most_recent_scanned_robot_age_turns: Age of scanned enemy, in turns.
        :param AA: List of actions that can be taken.
        :param terminal: Whether the state is terminal, meaning the episode has terminated naturally due to the
        dynamics of the environment. For example, the natural dynamics of the environment might terminate when the agent
        reaches a predefined goal state.
        :param truncated: Whether the state is truncated, meaning the episode has ended for some reason other than the
        natural dynamics of the environment. For example, imposing an artificial time limit on an episode might cause
        the episode to end without the agent in a predefined goal state.
        """

        super().__init__(
            i=None,
            AA=AA,
            terminal=terminal,
            truncated=truncated
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
        self.bullet_power_hit_others_cumulative = bullet_power_hit_others_cumulative
        self.bullet_power_missed_others = bullet_power_missed_others
        self.bullet_power_missed_others_cumulative = bullet_power_missed_others_cumulative
        self.bullet_power_missed_others_since_previous_hit = bullet_power_missed_others_since_previous_hit
        self.events = events
        self.previous_state: Optional[RobocodeState] = previous_state
        self.prior_state_different_location = prior_state_different_location
        self.most_recent_scanned_robot = most_recent_scanned_robot
        self.most_recent_scanned_robot_age_turns = most_recent_scanned_robot_age_turns

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        # don't pickle backreference to prior states, as pickling fails for such long recursion chains.
        state['previous_state'] = None
        state['prior_state_different_location'] = None

        return state


@rl_text(chapter='Feature Extractors', page=1)
class RobocodeFeatureExtractor(StateFeatureExtractor):
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
            environment: Environment
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        assert isinstance(environment, RobocodeEnvironment)

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls(
            environment=environment
        )

        return fex, unparsed_args

    def extracts_intercept(
            self
    ) -> bool:
        """
        Whether the feature extractor extracts an intercept (constant) term.

        :return: True if an intercept (constant) term is extracted and False otherwise.
        """

        return False

    def extract(
            self,
            states: List[MdpState],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param states: States.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature matrix (#states, #features).
        """

        x = np.array([
            self.get_feature_values(state)
            for state in states
            if isinstance(state, RobocodeState)
        ])

        if self.scale_features:
            x = self.feature_scaler.scale_features(
                x,
                refit_before_scaling=refit_scaler
            )

        return x

    def get_feature_values(
            self,
            state: RobocodeState
    ) -> List[float]:
        """
        Get feature values for a state.

        :param state: State.
        :return: List of floating-point feature values.
        """

        if state.most_recent_scanned_robot is None:
            most_recent_enemy_bearing_from_self = None
            most_recent_enemy_distance_from_self = None
            most_recent_scanned_robot_age_discount = None
        else:

            assert state.most_recent_scanned_robot_age_turns is not None

            # the bearing comes to us in radians. normalize to [0,360]. this bearing is relative to our heading, so
            # degrees from north to the enemy would be our heading plus this value.
            most_recent_enemy_bearing_from_self = self.normalize(math.degrees(state.most_recent_scanned_robot['bearing']))
            most_recent_enemy_distance_from_self = state.most_recent_scanned_robot['distance']
            most_recent_scanned_robot_age_discount = self.scanned_robot_decay ** state.most_recent_scanned_robot_age_turns

        # calculate continued distance ratio w.r.t. prior distance traveled
        if state.prior_state_different_location is not None:

            prior_distance_traveled = RobocodeFeatureExtractor.euclidean_distance(
                state.prior_state_different_location.x,
                state.prior_state_different_location.y,
                state.x,
                state.y
            )

            destination_x, destination_y = RobocodeFeatureExtractor.heading_destination(
                state.heading,
                state.x,
                state.y,
                prior_distance_traveled
            )

            continued_distance = RobocodeFeatureExtractor.euclidean_distance(
                state.prior_state_different_location.x,
                state.prior_state_different_location.y,
                destination_x,
                destination_y
            )

            continued_distance_ratio = continued_distance / prior_distance_traveled

        else:
            continued_distance_ratio = 0.0

        feature_values = [

            state.bullet_power_hit_self_cumulative,

            # we just hit a robot with our front
            1.0 if any(-90.0 < e['bearing'] < 90.0 for e in state.events.get('HitRobotEvent', [])) else 0.0,

            # we just hit a robot with our back
            1.0 if any(-90.0 > e['bearing'] > 90.0 for e in state.events.get('HitRobotEvent', [])) else 0.0,

            # distance ahead to boundary
            self.heading_distance_to_boundary(
                state.heading,
                state.x,
                state.y,
                state.battle_field_height,
                state.battle_field_width
            ),

            # distance behind to boundary
            self.heading_distance_to_boundary(
                self.normalize(state.heading - 180.0),
                state.x,
                state.y,
                state.battle_field_height,
                state.battle_field_width
            ),

            continued_distance_ratio,

            # bearing is clockwise-orthogonal to enemy
            0.0 if most_recent_enemy_bearing_from_self is None else
            most_recent_scanned_robot_age_discount * self.funnel(  # type: ignore[operator]
                self.get_shortest_degree_change(
                    state.heading,
                    self.normalize(state.heading + most_recent_enemy_bearing_from_self + 90.0)
                ),
                True,
                15.0
            ),

            # bearing is counterclockwise-orthogonal to enemy
            0.0 if most_recent_enemy_bearing_from_self is None else
            most_recent_scanned_robot_age_discount * self.funnel(  # type: ignore[operator]
                self.get_shortest_degree_change(
                    state.heading,
                    self.normalize(state.heading + most_recent_enemy_bearing_from_self - 90.0)
                ),
                True,
                15.0
            ),

            #  sigmoid_cw_ortho
            0.0 if most_recent_enemy_bearing_from_self is None else
            most_recent_scanned_robot_age_discount * self.sigmoid(  # type: ignore[operator]
                self.get_shortest_degree_change(
                    state.heading,
                    self.normalize(state.heading + most_recent_enemy_bearing_from_self + 90.0)
                ),
                15.0
            ),

            #  sigmoid_ccw_ortho
            0.0 if most_recent_enemy_bearing_from_self is None else
            most_recent_scanned_robot_age_discount * self.sigmoid(  # type: ignore[operator]
                self.get_shortest_degree_change(
                    state.heading,
                    self.normalize(state.heading + most_recent_enemy_bearing_from_self - 90.0)
                ),
                15.0
            ),

            state.bullet_power_missed_others_cumulative,

            # squashed lateral distance from radar to enemy
            0.0 if most_recent_enemy_bearing_from_self is None else
            most_recent_scanned_robot_age_discount * self.sigmoid(  # type: ignore[operator]
                self.get_lateral_distance(
                    -self.get_shortest_degree_change(
                        state.radar_heading,
                        self.normalize(state.heading + most_recent_enemy_bearing_from_self)
                    ),
                    most_recent_enemy_distance_from_self  # type: ignore[arg-type]
                ),
                20.0
            ),

            # squashed lateral distance from gun to enemy
            0.0 if most_recent_enemy_bearing_from_self is None else
            most_recent_scanned_robot_age_discount * self.sigmoid(  # type: ignore[operator]
                self.get_lateral_distance(
                    -self.get_shortest_degree_change(
                        state.gun_heading,
                        self.normalize(state.heading + most_recent_enemy_bearing_from_self)
                    ),
                    most_recent_enemy_distance_from_self  # type: ignore[arg-type]
                ),
                20.0
            ),

            state.bullet_power_hit_others_cumulative,

            # funnel_lat_dist:  funnel lateral distance to 0.0
            0.0 if most_recent_enemy_bearing_from_self is None else
            most_recent_scanned_robot_age_discount * self.funnel(  # type: ignore[operator]
                self.get_lateral_distance(
                    -self.get_shortest_degree_change(
                        state.gun_heading,
                        self.normalize(state.heading + most_recent_enemy_bearing_from_self)
                    ),
                    most_recent_enemy_distance_from_self  # type: ignore[arg-type]
                ),
                True,
                20.0
            )
        ]

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
    def compass_to_degrees(
            heading: float
    ) -> float:
        """
        Convert compass heading (0 @ north then clockwise) to standard trigonometric degrees (0 @ east and then
        counterclockwise).

        :param heading: Compass heading.
        :return: Trigonometric degrees.
        """

        return RobocodeFeatureExtractor.normalize(90.0 - heading)

    @staticmethod
    def compass_to_radians(
            heading: float
    ) -> float:
        """
        Convert compass heading (0 @ north then clockwise) to standard trigonometric radians (0 @ east and then
        counterclockwise).

        :param heading: Compass heading.
        :return: Trigonometric radians.
        """

        return math.radians(RobocodeFeatureExtractor.compass_to_degrees(heading))

    @staticmethod
    def compass_slope(
            heading: float
    ) -> float:
        """
        Get the slope of the compass heading.

        :param heading: Compass heading.
        :return: Slope.
        """

        return math.tan(RobocodeFeatureExtractor.compass_to_radians(heading))

    @staticmethod
    def compass_y_intercept(
            heading: float,
            x: float,
            y: float
    ) -> float:
        """
        Get the y-intercept of the compass heading.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :return: The y-intercept.
        """

        return y - RobocodeFeatureExtractor.compass_slope(heading) * x

    @staticmethod
    def top_intersect_x(
            heading: float,
            x: float,
            y: float,
            height: float
    ) -> float:
        """
        Get the x coordinate of the intersection of the compass heading with the top boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :param height: Height of boundary.
        :return: The x coordinate.
        """

        slope = RobocodeFeatureExtractor.compass_slope(heading)

        # if slope is zero, then we'll never intersect the top. return infinity. don't measure coverage for the case of
        # 0.0, as it's exceedingly difficult to generate the condition in a short test run.
        if slope == 0.0:  # pragma no cover
            intersect_x = np.inf
        else:
            y_intercept = RobocodeFeatureExtractor.compass_y_intercept(heading, x, y)
            intersect_x = (height - y_intercept) / slope

        return intersect_x

    @staticmethod
    def bottom_intersect_x(
            heading: float,
            x: float,
            y: float
    ) -> float:
        """
        Get the x coordinate of the intersection of the compass heading with the bottom boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :return: The x coordinate.
        """

        slope = RobocodeFeatureExtractor.compass_slope(heading)

        # if slope is zero, then we'll never intersect the top. return infinity. don't measure coverage for the case of
        # 0.0, as it's exceedingly difficult to generate the condition in a short test run.
        if slope == 0.0:  # pragma no cover
            intersect_x = np.inf
        else:
            y_intercept = RobocodeFeatureExtractor.compass_y_intercept(heading, x, y)
            intersect_x = -y_intercept / slope

        return intersect_x

    @staticmethod
    def right_intersect_y(
            heading: float,
            x: float,
            y: float,
            width: float
    ) -> float:
        """
        Get the y coordinate of the intersection of the compass heading with the right boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :param width: Width of boundary.
        :return: The y coordinate.
        """

        slope = RobocodeFeatureExtractor.compass_slope(heading)
        y_intercept = RobocodeFeatureExtractor.compass_y_intercept(heading, x, y)

        return slope * width + y_intercept

    @staticmethod
    def left_intersect_y(
            heading: float,
            x: float,
            y: float
    ) -> float:
        """
        Get the y coordinate of the intersection of the compass heading with the left boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :return: The y coordinate.
        """

        return RobocodeFeatureExtractor.compass_y_intercept(heading, x, y)

    @staticmethod
    def euclidean_distance(
            x_1: float,
            y_1: float,
            x_2: float,
            y_2: float
    ) -> float:
        """
        Get Euclidean distance between two points.

        :param x_1: First point's x coordinate.
        :param y_1: First point's y coordinate.
        :param x_2: Second point's x coordinate.
        :param y_2: Second point's y coordinate.
        :return: Euclidean distance.
        """

        return math.sqrt((x_2 - x_1) ** 2.0 + (y_2 - y_1) ** 2.0)

    @staticmethod
    def heading_distance_to_top(
            heading: float,
            x: float,
            y: float,
            height: float
    ) -> float:
        """
        Get heading distance to top boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :param height: Height of boundary.
        :return: Distance.
        """

        return RobocodeFeatureExtractor.euclidean_distance(
            x,
            y,
            RobocodeFeatureExtractor.top_intersect_x(heading, x, y, height),
            height
        )

    @staticmethod
    def heading_distance_to_right(
            heading: float,
            x: float,
            y: float,
            width: float
    ) -> float:
        """
        Get heading distance to top boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :param width: Width of boundary.
        :return: Distance.
        """

        return RobocodeFeatureExtractor.euclidean_distance(
            x,
            y,
            width,
            RobocodeFeatureExtractor.right_intersect_y(heading, x, y, width)
        )

    @staticmethod
    def heading_distance_to_bottom(
            heading: float,
            x: float,
            y: float
    ) -> float:
        """
        Get heading distance to top boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :return: Distance.
        """

        return RobocodeFeatureExtractor.euclidean_distance(
            x,
            y,
            RobocodeFeatureExtractor.bottom_intersect_x(heading, x, y),
            0.0
        )

    @staticmethod
    def heading_distance_to_left(
            heading: float,
            x: float,
            y: float
    ) -> float:
        """
        Get heading distance to top boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :return: Distance.
        """

        return RobocodeFeatureExtractor.euclidean_distance(
            x,
            y,
            0.0,
            RobocodeFeatureExtractor.left_intersect_y(heading, x, y)
        )

    @staticmethod
    def heading_distance_to_boundary(
            heading: float,
            x: float,
            y: float,
            height: float,
            width: float
    ) -> float:
        """
        Get heading distance to top boundary.

        :param heading: Compass heading.
        :param x: Current x position.
        :param y: Current y position.
        :param height: Height of boundary.
        :param width: Width of boundary.
        :return: Distance.
        """

        degrees = RobocodeFeatureExtractor.compass_to_degrees(heading)
        heading_quadrant = int(degrees / 90.0) + 1

        if heading_quadrant == 1:
            distance = min(
                RobocodeFeatureExtractor.heading_distance_to_top(heading, x, y, height),
                RobocodeFeatureExtractor.heading_distance_to_right(heading, x, y, width)
            )
        elif heading_quadrant == 2:
            distance = min(
                RobocodeFeatureExtractor.heading_distance_to_top(heading, x, y, height),
                RobocodeFeatureExtractor.heading_distance_to_left(heading, x, y)
            )
        elif heading_quadrant == 3:
            distance = min(
                RobocodeFeatureExtractor.heading_distance_to_left(heading, x, y),
                RobocodeFeatureExtractor.heading_distance_to_bottom(heading, x, y)
            )
        elif heading_quadrant == 4:
            distance = min(
                RobocodeFeatureExtractor.heading_distance_to_bottom(heading, x, y),
                RobocodeFeatureExtractor.heading_distance_to_right(heading, x, y, width)
            )
        else:  # pragma no cover
            raise ValueError(f'Invalid heading quadrant:  {heading_quadrant}')

        return distance

    @staticmethod
    def heading_destination(
            heading: float,
            current_x: float,
            current_y: float,
            distance: float
    ) -> Tuple[float, float]:
        """
        Get destination coordinates along a heading after traversing a given distance.

        :param heading: Current compass heading.
        :param current_x: Current x location.
        :param current_y: Current y location.
        :param distance: Distance to traverse.
        :return: 2-tuple of destination x-y coordinates.
        """

        radians = RobocodeFeatureExtractor.compass_to_radians(heading)

        delta_x = math.cos(radians) * distance
        delta_y = math.sin(radians) * distance

        return current_x + delta_x, current_y + delta_y

    @staticmethod
    def normalize(
            degrees: float
    ) -> float:
        """
        Normalize degrees to be [0, 360].

        :param degrees: Degrees.
        :return: Normalized degrees.
        """

        degrees = degrees % 360.0

        # the following should be impossible
        if degrees < 0 or degrees > 360:  # pragma no cover
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

        super().__init__(True)

        self.scanned_robot_decay = 0.75
        self.robot_actions = environment.robot_actions
        self.feature_scaler = StationaryFeatureScaler()


@rl_text(chapter='Actions', page=1)
class RobocodeAction(Enum):
    """
    Robocode action.
    """

    AHEAD = 'ahead'
    BACK = 'back'
    TURN_LEFT = 'turnLeft'
    TURN_RIGHT = 'turnRight'
    TURN_RADAR_LEFT = 'turnRadarLeft'
    TURN_RADAR_RIGHT = 'turnRadarRight'
    TURN_GUN_LEFT = 'turnGunLeft'
    TURN_GUN_RIGHT = 'turnGunRight'
    FIRE = 'fire'
