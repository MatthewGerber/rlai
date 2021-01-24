import math
import os
from argparse import ArgumentParser
from itertools import product
from typing import List, Tuple, Optional, Union

import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete, Box
from numpy.random import RandomState

from rlai.actions import Action, DiscretizedAction
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments
from rlai.value_estimation.function_approximation.models.feature_extraction import (
    StateActionInteractionFeatureExtractor,
    OneHotCategoricalFeatureInteracter,
    NonstationaryFeatureScaler
)


@rl_text(chapter='States', page=1)
class GymState(MdpState):
    """
    State of a Gym environment.
    """

    def __init__(
            self,
            environment,
            observation,
            agent: MdpAgent,
            terminal: bool,
    ):
        """
        Initialize the state.

        :param environment: Environment.
        :param observation: Observation.
        :param agent: Agent.
        :param terminal: Whether the state is terminal.
        """

        environment: Gym

        super().__init__(
            i=agent.pi.get_state_i(observation),
            AA=environment.actions,
            terminal=terminal
        )

        self.observation = observation


@rl_text(chapter='Environments', page=1)
class Gym(MdpEnvironment):
    """
    Generalized Gym environment. Any OpenAI Gym environment can be executed by supplying the appropriate identifier.
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
            '--gym-id',
            type=str,
            help='Initial seed count in each pit.'
        )

        parser.add_argument(
            '--continuous-action-discretization-resolution',
            type=float,
            help='Continuous-action discretization resolution.'
        )

        parser.add_argument(
            '--render-every-nth-episode',
            type=int,
            help='How often to render episodes into videos.'
        )

        parser.add_argument(
            '--video-directory',
            type=str,
            help='Local directory in which to save rendered videos. Must be an empty directory. Ignore to only display videos.'
        )

        parser.add_argument(
            '--force',
            action='store_true',
            help='Force the writing of videos into the video directory. Will overwrite/delete content in the directory.'
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

        gym_env = Gym(
            random_state=random_state,
            **vars(parsed_args)
        )

        return gym_env, unparsed_args

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: MdpAgent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance the state.

        :param state: State to advance.
        :param t: Time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of next state and reward.
        """

        # map discretized actions back to continuous space
        if isinstance(a, DiscretizedAction):
            gym_action = a.continuous_value
        else:
            gym_action = a.i

        observation, reward, done, _ = self.gym_native.step(action=gym_action)

        if self.render_current_episode:
            self.gym_native.render()

        next_state = GymState(
            environment=self,
            observation=observation,
            terminal=done,
            agent=agent
        )

        return next_state, Reward(i=None, r=reward)

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> GymState:
        """
        Reset the environment for a new run (episode).

        :param agent: Agent used to generate on-the-fly state identifiers.
        :return: Initial state.
        """

        super().reset_for_new_run(agent)

        # subtract 1 from number of resets in order to render the first episode
        if self.display_only_rendering:
            self.render_current_episode = (self.num_resets - 1) % self.render_every_nth_episode == 0

        observation = self.gym_native.reset()

        if self.render_current_episode:
            self.gym_native.render()

        self.state = GymState(
            environment=self,
            observation=observation,
            terminal=False,
            agent=agent
        )

        return self.state

    def close(
            self
    ):
        """
        Close the environment, releasing resources.
        """

        self.gym_native.close()

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            gym_id: str,
            continuous_action_discretization_resolution: Optional[float] = None,
            render_every_nth_episode: Optional[int] = None,
            video_directory: Optional[str] = None,
            force: bool = False
    ):
        """
        Initialize the environment.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param gym_id: Gym identifier. See https://gym.openai.com/envs for a list.
        :param continuous_action_discretization_resolution: A discretization resolution for continuous-action
        environments. Providing this value allows the environment to be used with discrete-action methods via
        discretization of the continuous-action dimensions.
        :param render_every_nth_episode: If passed, the environment will render an episode video per this value.
        :param video_directory: Directory in which to store rendered videos.
        :param force: Whether or not to force the writing of videos into the video directory. This will overwrite/delete
        content in the directory.
        """

        super().__init__(
            name=f'gym ({gym_id})',
            random_state=random_state,
            T=T
        )

        self.gym_native = gym.make(
            id=gym_id
        )

        # the native gym object uses the max value, so set it to something crazy huge if we're not given a T.
        self.gym_native._max_episode_steps = 999999999999 if self.T is None else self.T

        if continuous_action_discretization_resolution is not None and not isinstance(self.gym_native.action_space, Box):
            raise ValueError('Continuous-action discretization is only valid for Box action-space environments.')

        # set up rendering...either display-only or saved videos
        self.render_every_nth_episode = render_every_nth_episode
        self.display_only_rendering = False
        self.render_current_episode = False
        if self.render_every_nth_episode is not None:

            if self.render_every_nth_episode <= 0:
                raise ValueError('render_every_nth_episode must be > 0 if provided.')

            # display-only if we don't have a video directory
            if video_directory is None:
                self.display_only_rendering = True
                self.render_current_episode = True

            # saved videos via wrapper
            else:
                self.gym_native = gym.wrappers.Monitor(
                    env=self.gym_native,
                    directory=os.path.expanduser(video_directory),
                    video_callable=lambda episode_id: episode_id % self.render_every_nth_episode == 0,
                    force=force
                )

        self.gym_native.seed(random_state.randint(1000))

        # action space is already discrete. initialize n actions from it.
        if isinstance(self.gym_native.action_space, Discrete):
            self.actions = [
                Action(
                    i=i
                )
                for i in range(self.gym_native.action_space.n)
            ]

        # action space is continuous. discretize it.
        elif isinstance(self.gym_native.action_space, Box):

            box = self.gym_native.action_space

            # continuous n-dimensional action space with identical bounds on each dimension
            if len(box.shape) == 1:
                action_discretizations = [
                    np.linspace(low, high, math.ceil((high - low) / continuous_action_discretization_resolution))
                    for low, high in zip(box.low, box.high)
                ]
            else:  # pragma no cover
                raise ValueError(f'Unknown format of continuous action space:  {box}')

            self.actions = [
                DiscretizedAction(
                    i=i,
                    continuous_value=np.array(n_dim_action)
                )
                for i, n_dim_action in enumerate(product(*action_discretizations))
            ]

        else:  # pragma no cover
            raise ValueError(f'Unknown Gym action space type:  {type(self.gym_native.action_space)}')


@rl_text(chapter='Feature Extractors', page=1)
class CartpoleFeatureExtractor(StateActionInteractionFeatureExtractor):
    """
    A feature extractor for the OpenAI cartpole environment. This extractor, being based on the
    `StateActionInteractionFeatureExtractor`, directly extracts the fully interacted state-action feature matrix. It
    returns numpy.ndarray feature matrices, which are not compatible with the Patsy formula-based interface.
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
            environment: Gym
    ) -> Tuple[StateActionInteractionFeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        fex = CartpoleFeatureExtractor(
            environment=environment
        )

        return fex, unparsed_args

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            for_fitting: bool
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param for_fitting: Whether the extracted features will be used for fitting (True) or prediction (False).
        :return: State-feature pandas.DataFrame or numpy.ndarray.
        """

        self.check_state_and_action_lists(states, actions)

        X = np.array([
            np.append(state.observation, state.observation ** 2)
            for state in states
        ])

        X = self.feature_scaler.scale_features(X, for_fitting)

        contexts = [
            FeatureContext(
                cart_left_of_center=state.observation[0] < 0,
                cart_moving_left=state.observation[1] < 0,
                pole_left_of_vertical=state.observation[2] < 0,
                pole_rotating_left=state.observation[3] < 0
            )
            for state in states
        ]

        X = self.context_interacter.interact(X, contexts)

        X = self.interact(
            state_features=X,
            actions=actions
        )

        return X

    def get_feature_names(
            self
    ) -> List[str]:
        """
        Get names of extracted features.

        :return: List of feature names.
        """

        raise ValueError('Not implemented')

    def __init__(
            self,
            environment: Gym
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        if not isinstance(environment.gym_native.action_space, Discrete):  # pragma no cover
            raise ValueError('Expected a discrete action space, but did not get one.')

        if environment.gym_native.action_space.n != 2:  # pragma no cover
            raise ValueError('Expected two actions:  left and right')

        super().__init__(
            environment=environment,
            actions=[
                Action(
                    i=0,
                    name='left'
                ),
                Action(
                    i=1,
                    name='right'
                )
            ]
        )

        self.contexts = [
            FeatureContext(*context_bools)
            for context_bools in product([True, False], [True, False], [True, False], [True, False])
        ]

        self.context_interacter = OneHotCategoricalFeatureInteracter(self.contexts)

        self.feature_scaler = NonstationaryFeatureScaler(
            num_observations_refit_feature_scaler=2000,
            refit_history_length=10000,
            refit_weight_decay=0.9999
        )


class FeatureContext:
    """
    Categorical context within with a feature explains an outcome independently of its explanation in another context.
    This works quite similarly to traditional categorical interactions, except that they are specified programmatically
    by this class.
    """

    def __init__(
            self,
            cart_left_of_center: bool,
            cart_moving_left: bool,
            pole_left_of_vertical: bool,
            pole_rotating_left: bool
    ):
        """
        Initialize the context.

        :param cart_left_of_center: Left of center.
        :param cart_moving_left: Moving left.
        :param pole_left_of_vertical: Left of vertical.
        :param pole_rotating_left: Rotating left.
        """
        self.cart_left_of_center = cart_left_of_center
        self.cart_moving_left = cart_moving_left
        self.pole_left_of_vertical = pole_left_of_vertical
        self.pole_rotating_left = pole_rotating_left
        self.id = str(self)

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check equality.

        :param other: Other context.
        :return: True if equal and False otherwise.
        """

        other: FeatureContext

        return self.id == other.id

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check inequality.

        :param other: Other context.
        :return: True if unequal and False otherwise.
        """

        return not (self == other)

    def __hash__(
            self
    ) -> int:
        """
        Get hash code.

        :return: Hash code.
        """

        return hash(self.id)

    def __str__(
            self
    ) -> str:
        """
        Get string.

        :return: String.
        """

        return f'{self.cart_left_of_center}_{self.cart_moving_left}_{self.pole_left_of_vertical}_{self.pole_rotating_left}'
