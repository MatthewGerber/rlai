import logging
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

        # update discounted cumulative power that we've been hit with
        bullet_power_hit_self_cumulative = bullet_power_hit_self
        if self.previous_state is not None:
            turns_passed = client_dict['state']['time'] - self.previous_state.time
            bullet_power_hit_self_cumulative += self.previous_state.bullet_power_hit_self_cumulative * (0.9 ** turns_passed)

        # get most recent prior state that was at a different location than the current state. if there is no previous
        # state, then there is no such state.
        if self.previous_state is None:
            prior_state_different_location = None
        # if the previous state's location differs from the current location, then the previous state is what we want.
        elif self.previous_state.x != client_dict['state']['x'] or self.previous_state.y != client_dict['state']['y']:
            prior_state_different_location = self.previous_state
        # otherwise (if the previous and current state have the same location), then use the previous state's prior
        # state.
        else:
            prior_state_different_location = self.previous_state.prior_state_different_location

        # initialize the state
        state = RobocodeState(
            **client_dict['state'],
            bullet_power_hit_self=bullet_power_hit_self,
            bullet_power_hit_self_cumulative=bullet_power_hit_self_cumulative,
            bullet_power_hit_others=bullet_power_hit_others,
            bullet_power_missed=bullet_power_missed,
            bullet_power_missed_since_previous_hit=bullet_power_missed_since_previous_hit,
            events=event_type_events,
            previous_state=self.previous_state,
            prior_state_different_location=prior_state_different_location,
            AA=self.robot_actions,
            terminal=terminal
        )

        logging.debug(f'bullet_power_hit_self_cumulative:  {state.bullet_power_hit_self_cumulative}')

        # store bullet firing events so that we can pull out information related to them at a later time step (e.g.,
        # when they hit or miss). add the evaluation time step to each event. the bullet events have times associated
        # with them that are provided by the robocode engine, but those are robocode turns and there isn't always a
        # perfect 1:1 between robocode turns and evaluation time steps.
        self.bullet_id_fired_event.update({
            bullet_fired_event['bullet']['bulletId']: {
                **bullet_fired_event,
                'step': t
            }
            for bullet_fired_event in event_type_events.get('BulletFiredEvent', [])
        })

        # only issue a movement reward if we do not have an aiming reward. movement rewards are defined for every tick,
        # but aiming rewards are only nonzero when a bullet hits or misses. as aiming rewards are rarer, be sure to use
        # them whenever possible.
        aiming_reward_value = bullet_power_hit_others * 100.0 - bullet_power_missed
        if aiming_reward_value == 0:
            reward = RobocodeMovementReward(
                i=None,
                r=1.0 if bullet_power_hit_self == 0.0 else -10.0
            )
        else:
            reward = RobocodeAimingReward(
                i=None,
                r=aiming_reward_value,
                bullet_id_fired_event=self.bullet_id_fired_event,
                bullet_hit_events=bullet_hit_events,
                bullet_missed_events=bullet_missed_events
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
            (RobocodeAction.AHEAD, 25.0),
            (RobocodeAction.BACK, 25.0),
            (RobocodeAction.TURN_LEFT, 10.0),
            (RobocodeAction.TURN_RIGHT, 10.0),
            (RobocodeAction.TURN_RADAR_LEFT, 5.0),
            (RobocodeAction.TURN_RADAR_RIGHT, 5.0),
            (RobocodeAction.TURN_GUN_LEFT, 10.0),
            (RobocodeAction.TURN_GUN_RIGHT, 10.0),
            (RobocodeAction.FIRE, 1.0)
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
            previous_state,
            prior_state_different_location,
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
        :param previous_state: Previous state.
        :param prior_state_different_location: Most recent prior state at a different location than the current state.
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
        self.previous_state: RobocodeState = previous_state
        self.prior_state_different_location = prior_state_different_location

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state_dict = dict(self.__dict__)

        # don't pickle backreference to prior states, as pickling fails for such long recursion chains.
        state_dict['previous_state'] = None
        state_dict['prior_state_different_location'] = None

        return state_dict


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
            # the bearing comes to us in radians. normalize to [0,360]. this bearing is relative to our heading, so
            # degrees from north to the enemy would be our heading plus this value.
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

    def get_action_feature_names(
            self
    ) -> Dict[str, List[str]]:
        """
        Get names of actions and their associated feature names.

        :return: Dictionary of action names and their associated feature names.
        """

        return {

            action.name:

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'hit_by_bullet_power',
                'hit_robot',
                'enough_room',
                'continue_away',
                'funnel_cw_ortho',
                'funnel_ccw_ortho'
            ] if action.name == RobocodeAction.AHEAD else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'bullet_power_cum',
                'hit_robot',
                'enough_room',
                'continue_away',
                'funnel_cw_ortho',
                'funnel_ccw_ortho'
            ] if action.name == RobocodeAction.BACK else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'sigmoid_cw_ortho',
                'sigmoid_ccw_ortho'
            ] if action.name == RobocodeAction.TURN_LEFT else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'sigmoid_cw_ortho',
                'sigmoid_ccw_ortho'
            ] if action.name == RobocodeAction.TURN_RIGHT else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'sigmoid_lat_dist'
            ] if action.name == RobocodeAction.TURN_RADAR_LEFT else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'sigmoid_lat_dist'
            ] if action.name == RobocodeAction.TURN_RADAR_RIGHT else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'sigmoid_lat_dist'
            ] if action.name == RobocodeAction.TURN_GUN_LEFT else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'sigmoid_lat_dist'
            ] if action.name == RobocodeAction.TURN_GUN_RIGHT else

            [
                f'{action.name}_intercept',
                'enemy_bearing',
                'funnel_lat_dist'
            ]

            for action in self.robot_actions
        }

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

        if action_to_extract.name == RobocodeAction.AHEAD or action_to_extract.name == RobocodeAction.BACK:

            if action == action_to_extract:
                feature_values = [

                    # intercept
                    1.0,

                    # indicator (0/1):  we have a bearing on the enemy
                    0.0 if enemy_bearing_from_self is None else 1.0,

                    # discounted cumulative bullet power
                    state.bullet_power_hit_self_cumulative,

                    # we just hit a robot in the direction we're about to move
                    1.0 if (action.name == RobocodeAction.AHEAD and any(-90 < e['bearing'] < 90 for e in state.events.get('HitRobotEvent', []))) or
                           (action.name == RobocodeAction.BACK and any(-90 > e['bearing'] > 90 for e in state.events.get('HitRobotEvent', [])))
                    else 0.0,

                    # we have enough room to complete the move, plus a buffer.
                    1.0 if (action.name == RobocodeAction.AHEAD and action.value + 100.0 <= self.heading_distance_to_boundary(state.heading, state.x, state.y, state.battle_field_height, state.battle_field_width)) or
                           (action.name == RobocodeAction.BACK and action.value + 100.0 <= self.heading_distance_to_boundary(self.normalize(state.heading - 180.0), state.x, state.y, state.battle_field_height, state.battle_field_width))
                    else 0.0,

                    # the move will continue to take us farther from the most recent prior state whose location differs
                    # from the current location. we use this most recent prior state rather than the directly previous
                    # state because non-movement actions can intervene between movement and we don't want them to
                    # interfere with the feature value (they'll have the same location as the current state).
                    1.0 if state.prior_state_different_location is not None and RobocodeFeatureExtractor.euclidean_distance(
                        state.prior_state_different_location.x,
                        state.prior_state_different_location.y,
                        *RobocodeFeatureExtractor.heading_destination(state.heading if action.name == RobocodeAction.AHEAD else RobocodeFeatureExtractor.normalize(state.heading - 180.0), state.x, state.y, action.value)
                    ) > RobocodeFeatureExtractor.euclidean_distance(
                        state.prior_state_different_location.x,
                        state.prior_state_different_location.y,
                        state.x,
                        state.y
                    )
                    else 0.0,

                    # bearing is clockwise-orthogonal to enemy
                    0.0 if enemy_bearing_from_self is None else
                    self.funnel(
                        self.get_shortest_degree_change(
                            state.heading,
                            self.normalize(state.heading + enemy_bearing_from_self + 90.0)
                        ),
                        True,
                        15.0
                    ),

                    # bearing is counterclockwise-orthogonal to enemy
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
                feature_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        elif action_to_extract.name == RobocodeAction.TURN_LEFT or action_to_extract.name == RobocodeAction.TURN_RIGHT:

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

        elif action_to_extract.name == RobocodeAction.FIRE:

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

        # if slope is zero, then we'll never intersect the top. return infinity.
        if slope == 0.0:
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

        # if slope is zero, then we'll never intersect the top. return infinity.
        if slope == 0.0:
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

        logging.debug(f'Heading distance to boundary:  {distance}')

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

        self.robot_actions = environment.robot_actions
        self.most_recent_scanned_robot = None


@rl_text(chapter='Actions', page=1)
class RobocodeAction(Action):
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
