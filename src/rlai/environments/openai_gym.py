import math
import os
import warnings
from argparse import ArgumentParser
from itertools import product
from time import sleep
from typing import List, Tuple, Optional, Union, Dict

import gym
import numpy as np
from gym.envs.registration import EnvSpec
from gym.spaces import Discrete, Box
from gym.wrappers import TimeLimit
from numpy.random import RandomState

from rlai.actions import Action, DiscretizedAction, ContinuousMultiDimensionalAction
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import ContinuousMdpEnvironment
from rlai.meta import rl_text
from rlai.models.feature_extraction import NonstationaryFeatureScaler, FeatureExtractor
from rlai.q_S_A.function_approximation.models.feature_extraction import (
    StateActionInteractionFeatureExtractor,
    OneHotCategoricalFeatureInteracter
)
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments, ScatterPlot, ScatterPlotPosition
from rlai.v_S.function_approximation.models.feature_extraction import StateFeatureExtractor


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
class Gym(ContinuousMdpEnvironment):
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
            help='Gym identifier. See https://gym.openai.com/envs for a list of environments (e.g., CartPole-v1).'
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
            help='Pass this flag to force the writing of videos into the video directory. Will overwrite/delete content in the directory.'
        )

        parser.add_argument(
            '--steps-per-second',
            type=int,
            help='Number of steps per second when displaying videos.'
        )

        parser.add_argument(
            '--plot-environment',
            action='store_true',
            help='Pass this flag to plot environment values (e.g., state).'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[ContinuousMdpEnvironment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        gym_env = cls(
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
        # use continuous action values directly (need to be arrays)
        elif isinstance(a, ContinuousMultiDimensionalAction):
            gym_action = a.value
        # use discretized action indices
        else:
            gym_action = a.i

        observation, reward, done, _ = self.gym_native.step(action=gym_action)

        # override reward per environment
        if self.gym_id == 'LunarLanderContinuous-v2':
            reward = 2.0 - np.abs(observation[0:5]).sum()

        # call render if rendering manually
        if self.check_render_current_episode(True):
            self.gym_native.render()

        if self.check_render_current_episode(None):

            # sleep if we're restricting steps per second
            if self.steps_per_second is not None:
                sleep(1.0 / self.steps_per_second)

            if self.plot_environment:
                self.state_reward_scatter_plot.update(np.append(observation, reward))

        self.state = GymState(
            environment=self,
            observation=observation,
            terminal=done,
            agent=agent
        )

        return self.state, Reward(i=None, r=reward)

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

        if self.plot_environment:
            self.state_reward_scatter_plot.reset_y_range()

        observation = self.gym_native.reset()

        # call render if rendering manually
        if self.check_render_current_episode(True):
            self.gym_native.render()

        self.state = GymState(
            environment=self,
            observation=observation,
            terminal=False,
            agent=agent
        )

        return self.state

    def check_render_current_episode(
            self,
            render_manually: Optional[bool]
    ) -> bool:
        """
        Check whether the current episode is to be rendered.

        :param render_manually: Wether the rendering will be done manually with calls to the render function or automatically
        as a result of saving videos via the monitor. Pass None to check whether the episode should be rendered,
        regardless of how the rendering will be done.
        :return: True if rendered and False otherwise.
        """

        # subtract 1 from number of resets to render first episode
        check_result = self.render_every_nth_episode is not None and (self.num_resets - 1) % self.render_every_nth_episode == 0

        if render_manually is not None:
            if render_manually:
                check_result = check_result and self.video_directory is None
            else:
                check_result = check_result and self.video_directory is not None

        return check_result

    def close(
            self
    ):
        """
        Close the environment, releasing resources.
        """

        self.gym_native.close()

        if self.state_reward_scatter_plot is not None:
            self.state_reward_scatter_plot.close()

    def init_gym_native(
            self
    ) -> Union[EnvSpec, TimeLimit]:
        """
        Initialize the native Gym environment object.

        :return: Either a native Gym environment or a wrapped native Gym environment.
        """

        gym_native = gym.make(
            id=self.gym_id
        )

        # the native gym object uses the max value, so set it to something crazy huge if we're not given a T.
        gym_native._max_episode_steps = 999999999999 if self.T is None else self.T

        # save videos via wrapper if we have a video directory
        if self.render_every_nth_episode is not None and self.video_directory is not None:
            try:
                gym_native = gym.wrappers.Monitor(
                    env=gym_native,
                    directory=os.path.expanduser(self.video_directory),
                    video_callable=lambda episode_id: episode_id % self.render_every_nth_episode == 0,
                    force=self.force
                )

            # pickled checkpoints can come from another os where the video directory is valid, but the directory might
            # not be valid on the current os. warn about permission errors and skip video saving.
            except PermissionError as ex:
                warnings.warn(f'Permission error when initializing OpenAI Gym monitor. Videos will not be saved. Error:  {ex}')

        gym_native.seed(self.random_state.randint(1000))

        return gym_native

    def get_state_space_dimensionality(
            self
    ) -> int:
        """
        Get the dimensionality of the state space.

        :return: Number of dimensions.
        """

        return self.gym_native.observation_space.shape[0]

    def get_state_dimension_names(
            self
    ) -> List[str]:
        """
        Get names of state dimensions.

        :return: List of names.
        """

        if self.gym_id == 'LunarLanderContinuous-v2':
            names = [
                'posX',
                'posY',
                'velX',
                'velY',
                'ang',
                'angV',
                'leg1Con',
                'leg2Con'
            ]
        else:  # pragma no cover
            warnings.warn(f'The state dimension names for {self.gym_id} are unknown. Defaulting to numbers.')
            names = [str(x) for x in range(0, self.get_state_space_dimensionality())]

        return names

    def get_action_space_dimensionality(
            self
    ) -> int:
        """
        Get the dimensionality of the action space.

        :return: Number of dimensions.
        """

        return self.gym_native.action_space.shape[0]

    def get_action_dimension_names(
            self
    ) -> List[str]:
        """
        Get action names.

        :return: List of names.
        """

        if self.gym_id == 'LunarLanderContinuous-v2':
            names = [
                'main',
                'side'
            ]
        else:  # pragma no cover
            warnings.warn(f'The action names for {self.gym_id} are unknown. Defaulting to numbers.')
            names = [str(x) for x in range(0, self.get_action_space_dimensionality())]

        return names

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            gym_id: str,
            continuous_action_discretization_resolution: Optional[float] = None,
            render_every_nth_episode: Optional[int] = None,
            video_directory: Optional[str] = None,
            force: bool = False,
            steps_per_second: Optional[int] = None,
            plot_environment: bool = False
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
        :param steps_per_second: Number of steps per second when displaying videos.
        :param plot_environment: Whether or not to plot the environment.
        """

        super().__init__(
            name=f'gym ({gym_id})',
            random_state=random_state,
            T=T
        )

        self.gym_id = gym_id
        self.continuous_action_discretization_resolution = continuous_action_discretization_resolution
        self.render_every_nth_episode = render_every_nth_episode
        if self.render_every_nth_episode is not None and self.render_every_nth_episode <= 0:
            raise ValueError('render_every_nth_episode must be > 0 if provided.')

        self.video_directory = video_directory
        self.force = force
        self.steps_per_second = steps_per_second
        self.plot_environment = plot_environment
        self.state_reward_scatter_plot = None
        if self.plot_environment:
            self.state_reward_scatter_plot = ScatterPlot(
                f'{self.gym_id}:  State and Reward',
                self.get_state_dimension_names() + ['reward'],
                None
            )

        self.gym_native = self.init_gym_native()

        if self.continuous_action_discretization_resolution is not None and not isinstance(self.gym_native.action_space, Box):
            raise ValueError('Continuous-action discretization is only valid for Box action-space environments.')

        # action space is already discrete:  initialize n actions from it.
        if isinstance(self.gym_native.action_space, Discrete):
            self.actions = [
                Action(
                    i=i
                )
                for i in range(self.gym_native.action_space.n)
            ]

        # action space is continuous and we lack a discretization resolution:  initializee a single, multi-dimensional
        # action including the min and max values of the dimensions.
        elif isinstance(self.gym_native.action_space, Box) and self.continuous_action_discretization_resolution is None:
            self.actions = [
                ContinuousMultiDimensionalAction(
                    value=None,
                    min_values=self.gym_native.action_space.low,
                    max_values=self.gym_native.action_space.high
                )
            ]

        # action space is continuous and we have a discretization resolution:  discretize it. this is generally not a
        # great approach, as it results in high-dimensional action spaces.
        elif isinstance(self.gym_native.action_space, Box) and self.continuous_action_discretization_resolution is not None:

            box = self.gym_native.action_space

            # continuous n-dimensional action space with identical bounds on each dimension
            if len(box.shape) == 1:
                action_discretizations = [
                    np.linspace(low, high, math.ceil((high - low) / self.continuous_action_discretization_resolution))
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

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        # the native gym environment cannot be pickled. blank it out.
        state['gym_native'] = None

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set the state dictionary.

        :param state: State dictionary.
        """

        self.__dict__ = state

        self.gym_native = self.init_gym_native()


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

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls(
            environment=environment
        )

        return fex, unparsed_args

    def extract(
            self,
            states: List[MdpState],
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

        X = np.array([
            np.append(state.observation, state.observation ** 2)
            for state in states
        ])

        X = self.feature_scaler.scale_features(X, for_fitting)

        contexts = [
            CartpoleFeatureContext(
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
            CartpoleFeatureContext(*context_bools)
            for context_bools in product([True, False], [True, False], [True, False], [True, False])
        ]

        self.context_interacter = OneHotCategoricalFeatureInteracter(self.contexts)

        self.feature_scaler = NonstationaryFeatureScaler(
            num_observations_refit_feature_scaler=2000,
            refit_history_length=100000,
            refit_weight_decay=0.99999
        )


class CartpoleFeatureContext:
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

        other: CartpoleFeatureContext

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


@rl_text(chapter='Feature Extractors', page=1)
class ContinuousFeatureExtractor(StateFeatureExtractor):
    """
    A feature extractor for continuous OpenAI environments.
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
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls()

        return fex, unparsed_args

    def extract(
            self,
            state: GymState,
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :return: State-feature vector.
        """

        return state.observation

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """

        super().__init__()
