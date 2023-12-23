import math
import os
import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from itertools import product
from time import sleep
from typing import List, Tuple, Optional, Union, Dict

import gymnasium
import numpy as np
from PyQt6.QtWidgets import QApplication
from gymnasium.spaces import Discrete, Box
from gymnasium.wrappers import TimeLimit, RecordVideo
from numpy.random import RandomState

from rlai.core import (
    Reward,
    Action,
    DiscretizedAction,
    ContinuousMultiDimensionalAction,
    MdpState,
    MdpAgent,
    Environment
)
from rlai.core.environments.mdp import ContinuousMdpEnvironment
from rlai.gpi.state_action_value.function_approximation.models.feature_extraction import (
    StateActionInteractionFeatureExtractor
)
from rlai.meta import rl_text
from rlai.models.feature_extraction import (
    FeatureExtractor,
    OneHotCategoricalFeatureInteracter,
    OneHotCategory,
    StationaryFeatureScaler
)
from rlai.state_value.function_approximation.models.feature_extraction import (
    StateFeatureExtractor,
    OneHotStateSegmentFeatureInteracter
)
from rlai.utils import parse_arguments, ScatterPlot


@rl_text(chapter='States', page=1)
class GymState(MdpState):
    """
    State of a Gym environment.
    """

    def __init__(
            self,
            environment: 'Gym',
            observation: np.ndarray,
            agent: MdpAgent,
            terminal: bool,
            truncated: bool
    ):
        """
        Initialize the state.

        :param environment: Environment.
        :param observation: Observation.
        :param agent: Agent.
        :param terminal: Whether the state is terminal, meaning the episode has terminated naturally due to the
        dynamics of the environment. For example, the natural dynamics of the environment might terminate when the agent
        reaches a predefined goal state.
        :param truncated: Whether the state is truncated, meaning the episode has ended for some reason other than the
        natural dynamics of the environment. For example, imposing an artificial time limit on an episode might cause
        the episode to end without the agent in a predefined goal state.
        """

        super().__init__(
            i=agent.pi.get_state_i(observation),
            AA=environment.actions,
            terminal=terminal,
            truncated=truncated
        )

        self.observation = observation

    def __str__(self) -> str:
        """
        Get string.

        :return: String.
        """

        return f'{self.observation}'


@rl_text(chapter='Environments', page=1)
class Gym(ContinuousMdpEnvironment):
    """
    Generalized Gym environment. Any Gym environment can be executed by supplying the appropriate identifier.
    """

    LLC_V2 = 'LunarLanderContinuous-v2'
    MCC_V0 = 'MountainCarContinuous-v0'
    SWIMMER_V2 = 'Swimmer-v2'
    CARTPOLE_V1 = 'CartPole-v1'

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
            help='Gym identifier. See https://gymnasium.farama.org for a list of environments (e.g., CartPole-v1).'
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
    ) -> Tuple[Environment, List[str]]:
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

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            gym_id: str,
            continuous_action_discretization_resolution: Optional[float] = None,
            render_every_nth_episode: Optional[int] = None,
            video_directory: Optional[str] = None,
            steps_per_second: Optional[int] = None,
            plot_environment: bool = False
    ):
        """
        Initialize the environment.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param gym_id: Gym identifier. See https://gymnasium.farama.org for a list.
        :param continuous_action_discretization_resolution: A discretization resolution for continuous-action
        environments. Providing this value allows the environment to be used with discrete-action methods via
        discretization of the continuous-action dimensions.
        :param render_every_nth_episode: If passed, the environment will render an episode video per this value.
        :param video_directory: Directory in which to store rendered videos.
        :param steps_per_second: Number of steps per second when displaying videos.
        :param plot_environment: Whether to plot the environment.
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
        self.steps_per_second = steps_per_second
        self.gym_native = self.init_gym_native()

        if self.gym_id == Gym.LLC_V2:
            self.gym_extender = ContinuousLunarLander(self.gym_native)
        elif self.gym_id == Gym.MCC_V0:
            self.gym_extender = ContinuousMountainCar(self.gym_native)
        elif self.gym_id == Gym.CARTPOLE_V1:
            self.gym_extender = Cartpole(self.gym_native)
        else:
            self.gym_extender: Optional[GymExtender] = None

        self.plot_environment = plot_environment
        self.state_reward_scatter_plot = None
        if self.plot_environment:
            self.state_reward_scatter_plot = ScatterPlot(
                f'{self.gym_id}:  State and Reward',
                self.get_state_dimension_names() + ['reward'],
                None
            )

        if (
            self.continuous_action_discretization_resolution is not None and
            not isinstance(self.gym_native.action_space, Box)
        ):
            raise ValueError('Continuous-action discretization is only valid for Box action-space environments.')

        action_space = self.gym_native.action_space

        # action space is already discrete:  initialize n actions from it.
        if isinstance(action_space, Discrete):
            self.actions = [
                Action(
                    i=i,
                    name=name
                )
                for i, name in zip(
                    range(action_space.n),
                    [None] * action_space.n if self.gym_extender is None
                    else self.gym_extender.get_action_names
                )
            ]

        # action space is continuous, and we lack a discretization resolution:  initialize a single, multidimensional
        # action including the min and max values of the dimensions. a policy gradient approach will be required.
        elif isinstance(action_space, Box) and self.continuous_action_discretization_resolution is None:
            self.actions = [
                ContinuousMultiDimensionalAction(
                    value=None,
                    min_values=action_space.low,
                    max_values=action_space.high
                )
            ]

        # action space is continuous, and we have a discretization resolution:  discretize it. this is generally not a
        # great approach, as it results in high-dimensional action spaces. but here goes.
        elif isinstance(action_space, Box) and self.continuous_action_discretization_resolution is not None:

            # continuous n-dimensional action space with identical bounds on each dimension
            if len(action_space.shape) == 1:
                action_discretizations = [
                    np.linspace(low, high, math.ceil((high - low) / self.continuous_action_discretization_resolution))
                    for low, high in zip(action_space.low, action_space.high)
                ]
            else:  # pragma no cover
                raise ValueError(f'Unknown format of continuous action space:  {action_space}')

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

        assert isinstance(state, GymState)

        # map discretized actions back to continuous space
        if isinstance(a, DiscretizedAction):
            gym_action = a.continuous_value

        # use continuous action values (which are vectors) directly
        elif isinstance(a, ContinuousMultiDimensionalAction):
            gym_action = a.value

        # use discretized action indices
        else:
            gym_action = a.i

        if self.gym_extender is not None:
            gym_action = self.gym_extender.get_action_to_step(gym_action)

        observation, reward, terminated, truncated, _ = self.gym_native.step(action=gym_action)

        if self.gym_extender is not None:
            observation = self.gym_extender.get_post_step_observation(observation)
            reward = self.gym_extender.get_reward(float(reward), observation, terminated, truncated)

        # call render if rendering manually
        if self.check_render_current_episode(True):
            self.gym_native.render()

        if self.check_render_current_episode(None):

            # sleep if we're restricting steps per second
            if self.steps_per_second is not None:
                sleep(1.0 / self.steps_per_second)

            if self.plot_environment:
                self.state_reward_scatter_plot.update(np.append(observation, reward))

            # swimmer is a non-qt environment, so we need to process qt events manually.
            if self.gym_id == Gym.SWIMMER_V2:
                QApplication.processEvents()

        self.state = GymState(
            environment=self,
            observation=observation,
            agent=agent,
            terminal=terminated,
            truncated=truncated
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

        observation, _ = self.gym_native.reset()

        if self.gym_extender is not None:
            observation = self.gym_extender.get_reset_observation(observation)

        # call render if rendering manually
        if self.check_render_current_episode(True):
            self.gym_native.render()

        self.state = GymState(
            environment=self,
            observation=observation,
            agent=agent,
            terminal=False,
            truncated=False
        )

        return self.state

    def check_render_current_episode(
            self,
            render_manually: Optional[bool]
    ) -> bool:
        """
        Check whether the current episode is to be rendered.

        :param render_manually: Whether the rendering will be done manually with calls to the render function or
        automatically as a result of saving videos via the monitor. Pass None to check whether the episode should be
        rendered, regardless of how the rendering will be done.
        :return: True if rendered and False otherwise.
        """

        # subtract 1 from number of resets to render first episode
        check_result = (
            self.render_every_nth_episode is not None and
            (self.num_resets - 1) % self.render_every_nth_episode == 0
        )

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
    ) -> Union[TimeLimit, RecordVideo]:
        """
        Initialize the native Gym environment object.

        :return: Either a native Gym environment or a wrapped native Gym environment.
        """

        record_video = self.render_every_nth_episode is not None and self.video_directory is not None

        gym_native = gymnasium.make(
            id=self.gym_id,
            max_episode_steps=self.T,
            render_mode=(

                # emit an rgb array for the step's frame if we're recording video
                'rgb_array' if record_video

                # emit human-scaled rendering if we're not recording a video but we need to render
                else 'human' if self.render_every_nth_episode

                # otherwise, do not render.
                else None
            )
        )

        # save videos via wrapper if we are recording
        if record_video:
            try:
                gym_native = RecordVideo(
                    env=gym_native,
                    video_folder=os.path.expanduser(self.video_directory),
                    episode_trigger=lambda episode_id: episode_id % self.render_every_nth_episode == 0
                )

            # pickled checkpoints can come from another os where the video directory is valid, but the directory might
            # not be valid on the current os. warn about permission errors and skip video saving.
            except PermissionError as ex:
                warnings.warn(f'Permission error when initializing Gym monitor. Videos will not be saved. Error:  {ex}')

        gym_native.reset(seed=self.random_state.randint(1000))

        return gym_native

    def get_state_space_dimensionality(
            self
    ) -> int:
        """
        Get the dimensionality of the state space.

        :return: Number of dimensions.
        """

        if self.gym_extender is None:
            dimensionality = self.gym_native.observation_space.shape[0]
        else:
            dimensionality = len(self.gym_extender.get_state_dimension_names())

        return dimensionality

    def get_state_dimension_names(
            self
    ) -> List[str]:
        """
        Get names of state dimensions.

        :return: List of names.
        """

        if self.gym_extender is None:
            warnings.warn(f'The state dimension names for {self.gym_id} are unknown. Defaulting to numbers.')
            names = [str(x) for x in range(0, self.get_state_space_dimensionality())]
        else:
            names = self.gym_extender.get_state_dimension_names()

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
        Get names of action dimensions.

        :return: List of names.
        """

        assert isinstance(self.gym_extender, ContinuousActionGym)

        if self.gym_extender is None:
            warnings.warn(f'The action dimension names for {self.gym_id} are unknown. Defaulting to numbers.')
            names = [str(x) for x in range(0, self.get_action_space_dimensionality())]
        else:
            names = self.gym_extender.get_action_dimension_names()

        return names


class GymExtender(ABC):
    """
    Abstract class for standard Gym environments. This provides a standard interface for customizing the behavior of
    environments.
    """

    def __init__(
            self,
            gym_native: Union[TimeLimit, RecordVideo]
    ):
        self.gym_native = gym_native

    @abstractmethod
    def get_state_dimension_names(
            self
    ) -> List[str]:
        """
        Get state-dimension names.

        :return: List of names.
        """

    @abstractmethod
    def get_reset_observation(
            self,
            observation: np.ndarray
    ) -> np.ndarray:
        """
        Get observation for reset.

        :param observation: Observation.
        :return: Observation.
        """

    @abstractmethod
    def get_action_to_step(
            self,
            action: np.ndarray
    ) -> np.ndarray:
        """
        Get action to step.

        :param action: Gym action.
        :return: Action to step.
        """

    @abstractmethod
    def get_post_step_observation(
            self,
            observation: np.ndarray
    ) -> np.ndarray:
        """
        Get observation resulting from a step.

        :param observation: Observation.
        :return: Observation resulting from a step.
        """

    @abstractmethod
    def get_reward(
            self,
            reward: float,
            observation: np.ndarray,
            terminated: bool,
            truncated: bool
    ) -> float:
        """
        Get reward.

        :param reward: Reward specified by the native Gym environment.
        :param observation: Observation.
        :param terminated: Terminated.
        :param truncated: Truncated.
        :return: Reward.
        """


class DiscreteActionGym(GymExtender, ABC):
    """
    Discrete-action Gym environment.
    """

    @abstractmethod
    def get_action_names(
            self
    ) -> List[Optional[str]]:
        """
        Get action-dimension names.

        :return: List of names.
        """


class ContinuousActionGym(GymExtender, ABC):
    """
    Continuous-action Gym environment.
    """

    @abstractmethod
    def get_action_dimension_names(
            self
    ) -> List[str]:
        """
        Get action-dimension names.

        :return: List of names.
        """


@rl_text(chapter='Feature Extractors', page=1)
class ContinuousFeatureExtractor(StateFeatureExtractor):
    """
    A feature extractor for continuous Gym environments. Extracts a scaled (standardized) version of the Gym state
    observation.
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
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: Scaled (standardized) state-feature vector.
        """

        return self.feature_scaler.scale_features(
            np.array([state.observation]),
            refit_before_scaling=refit_scaler
        )[0]

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """

        super().__init__()

        self.feature_scaler = StationaryFeatureScaler()


@rl_text(chapter='Feature Extractors', page=1)
class SignedCodingFeatureExtractor(ContinuousFeatureExtractor):
    """
    Signed-coding feature extractor. Forms a category from the conjunction of all state-feature signs and then places
    the continuous feature vector into its associated category. Works for all continuous-valued state spaces in Gym.
    """

    def extract(
            self,
            state: GymState,
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is only
        appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature vector.
        """

        state_matrix = np.array([state.observation])

        if self.state_category_interacter is None:
            self.state_category_interacter = OneHotStateSegmentFeatureInteracter({
                dimension: [0.0]
                for dimension in range(state_matrix.shape[1])
            })

        # extract and encode feature values
        scaled_feature_vector = super().extract(state, refit_scaler)
        interacted_feature_vector = self.state_category_interacter.interact(
            state_matrix,
            np.array([scaled_feature_vector])
        )[0]

        return interacted_feature_vector

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """

        super().__init__()

        # this is a generic feature extractor for all gym environments. as such, we don't know the dimensionality of the
        # state space until the first call to extract. do lazy-init here.
        self.state_category_interacter = None


class Cartpole(DiscreteActionGym):

    def get_action_names(self) -> List[Optional[str]]:

        return ['push-left', 'push-right']

    def get_state_dimension_names(self) -> List[str]:
        return [
            'pos',
            'vel',
            'ang',
            'angV'
        ]

    def get_reset_observation(self, observation: np.ndarray) -> np.ndarray:
        pass

    def get_action_to_step(self, action: np.ndarray) -> np.ndarray:
        pass

    def get_post_step_observation(self, observation: np.ndarray) -> np.ndarray:
        pass

    def get_reward(self, reward: float, observation: np.ndarray, terminated: bool, truncated: bool) -> float:

        return np.exp(
            -(
                np.abs([
                    observation[0],
                    observation[2] * 7.5,  # equalize the angle's scale with the position's scale
                ]).sum()
            )
        )


@rl_text(chapter='Feature Extractors', page=1)
class CartpoleFeatureExtractor(StateActionInteractionFeatureExtractor):
    """
    A feature extractor for the Gym cartpole environment. This extractor, being based on the
    `StateActionInteractionFeatureExtractor`, directly extracts the fully interacted state-action feature matrix. It
    returns numpy.ndarray feature matrices, which are not compatible with the Patsy formula-based interface. Lastly, and
    importantly, it adds a constant term to the state-feature vector before all interactions, which results in a
    separate intercept term being present for each state segment and action combination. The function approximator
    should not add its own intercept term.
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

        fex = cls(
            environment=environment
        )

        return fex, unparsed_args

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature numpy.ndarray.
        """

        states: List[GymState]

        self.check_state_and_action_lists(states, actions)

        # get the raw state matrix
        state_matrix = np.array([
            state.observation
            for state in states
        ])

        # extract and scale features for each state vector
        state_feature_matrix = np.array([
            np.append(state_vector, [1.0])
            for state_vector in state_matrix
        ])
        state_scaled_feature_matrix = self.feature_scaler.scale_features(state_feature_matrix, refit_scaler)

        # interact the feature matrix with its state-segment indicators
        state_category_feature_matrix = self.state_segment_interacter.interact(
            state_matrix,
            state_scaled_feature_matrix
        )

        # interact features per action
        state_action_feature_matrix = self.interact(
            state_features=state_category_feature_matrix,
            actions=actions
        )

        return state_action_feature_matrix

    def __init__(
            self,
            environment: Gym
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        action_space = environment.gym_native.action_space

        if not isinstance(action_space, Discrete):  # pragma no cover
            raise ValueError(f'Expected a {Discrete} action space.')

        if action_space.n != 2:  # pragma no cover
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

        self.state_segment_interacter = OneHotStateSegmentFeatureInteracter({

            # cart position is [-2.4, 2.4]
            0: [-1.2, 0, 1.2],

            # cart velocity is (-inf, inf) but typical values are in [-2.0, 2.0]
            1: [-1.5, 0.0, 1.5],

            # pole angle is [-.2095, .2095]
            2: [-0.1, 0.0, 0.1],

            # pole angle velocity is (-inf, inf) but typical values are in [-2.0, 2.0]
            3: [-1.5, 0.0, 1.5]
        })

        self.feature_scaler = StationaryFeatureScaler()


class ContinuousMountainCar(ContinuousActionGym):
    """
    Continuous mountain car.
    """

    TROUGH_X_POS = -0.5
    GOAL_X_POS = 0.45
    MAX_FUEL_USE_PER_STEP = 1.0 / 300.0

    def __init__(
            self,
            gym_native: Union[TimeLimit, RecordVideo]
    ):
        super().__init__(
            gym_native
        )

        self.fuel_level = 1.0
        self.mcc_curr_goal_x_pos = self.TROUGH_X_POS + 0.1

    def get_action_dimension_names(self) -> List[str]:

        return [
            'throttle'
        ]

    def get_state_dimension_names(self) -> List[str]:

        return [
            'position',
            'velocity',
            'fuel_level'
        ]

    def get_reset_observation(self, observation: np.ndarray) -> np.ndarray:

        self.fuel_level = 1.0
        observation = np.append(observation, self.fuel_level)

        return observation

    def get_action_to_step(
            self,
            action: np.ndarray
    ) -> np.ndarray:

        throttle = action[0]
        required_fuel = self.MAX_FUEL_USE_PER_STEP * abs(throttle)
        action_to_step = action.copy()
        if required_fuel > self.fuel_level:
            action_to_step[:] *= self.fuel_level / required_fuel
            self.fuel_level = 0.0
        else:
            self.fuel_level -= required_fuel

        return action_to_step

    def get_post_step_observation(
            self,
            observation: np.ndarray
    ) -> np.ndarray:

        return np.append(observation, self.fuel_level)

    def get_reward(
            self,
            reward: float,
            observation: np.ndarray,
            terminated: bool,
            truncated: bool
    ) -> float:

        reward = 0.0

        # calculate fraction to goal state
        curr_distance = observation[0] - self.TROUGH_X_POS
        goal_distance = self.mcc_curr_goal_x_pos - self.TROUGH_X_POS
        fraction_to_goal = curr_distance / goal_distance
        if fraction_to_goal >= 1.0:

            # increment goal up to the final goal
            self.mcc_curr_goal_x_pos = min(self.GOAL_X_POS, self.mcc_curr_goal_x_pos + 0.05)

            # mark state and stats recorder as done. must manually mark stats recorder to allow premature reset.
            terminated = True
            if hasattr(self.gym_native, 'stats_recorder'):
                self.gym_native.stats_recorder.done = terminated

            reward = curr_distance + self.fuel_level

        return reward


@rl_text(chapter='Feature Extractors', page=1)
class ContinuousMountainCarFeatureExtractor(ContinuousFeatureExtractor):
    """
    Feature extractor for the continuous mountain car environment.
    """

    def extract(
            self,
            state: GymState,
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature vector.
        """

        # extract raw feature values
        scaled_feature_matrix = super().extract(state, refit_scaler)
        interacted_feature_vector = self.state_category_interacter.interact(
            np.array([state.observation]),
            np.array([scaled_feature_matrix])
        )[0]

        return interacted_feature_vector

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """

        super().__init__()

        # interact features with relevant state categories
        self.state_category_interacter = OneHotStateSegmentFeatureInteracter({

            # shift the x-location midpoint to the bottom of the trough
            0: [ContinuousMountainCar.TROUGH_X_POS],

            # velocity switches at zero
            1: [0.0],

            # fuel bottoms out at zero
            2: [0.0000001]
        })


class ContinuousLunarLander(ContinuousActionGym):
    """
    Continuous lunar lander.
    """

    MAIN_MAX_FUEL_USE_PER_STEP = 1.0 / 300.0
    SIDE_MAX_FUEL_USE_PER_STEP = 1.0 / 600.0

    def __init__(
            self,
            gym_native: Union[TimeLimit, RecordVideo]
    ):
        super().__init__(
            gym_native
        )

        self.fuel_level = 1.0

    def get_action_dimension_names(self) -> List[str]:

        return [
            'main',
            'side'
        ]

    def get_state_dimension_names(self) -> List[str]:

        return [
            'posX',
            'posY',
            'velX',
            'velY',
            'ang',
            'angV',
            'leg1Con',
            'leg2Con',
            'fuel_level'
        ]

    def get_reset_observation(self, observation: np.ndarray) -> np.ndarray:

        self.fuel_level = 1.0
        observation = np.append(observation, self.fuel_level)

        return observation

    def get_action_to_step(
            self,
            action: np.ndarray
    ) -> np.ndarray:

        main_throttle, side_throttle = action[:]
        if main_throttle >= 0.0:
            required_main_fuel = self.MAIN_MAX_FUEL_USE_PER_STEP * (0.5 + 0.5 * main_throttle)
        else:
            required_main_fuel = 0.0

        if abs(side_throttle) >= 0.5:
            required_side_fuel = self.SIDE_MAX_FUEL_USE_PER_STEP * abs(side_throttle)
        else:
            required_side_fuel = 0.0

        action_to_step = action.copy()
        required_total_fuel = required_main_fuel + required_side_fuel
        if required_total_fuel > self.fuel_level:
            action_to_step[:] *= self.fuel_level / required_total_fuel
            self.fuel_level = 0.0
        else:
            self.fuel_level -= required_total_fuel

        return action_to_step

    def get_post_step_observation(
            self,
            observation: np.ndarray
    ) -> np.ndarray:

        return np.append(observation, self.fuel_level)

    def get_reward(
            self,
            reward: float,
            observation: np.ndarray,
            terminated: bool,
            truncated: bool
    ) -> float:

        reward = 0.0

        if terminated:

            # the ideal state is zeros across position/movement
            state_reward = -np.abs(observation[0:6]).sum()

            # reward for remaining fuel, but only if the state is good. rewarding for remaining fuel unconditionally
            # can cause the agent to veer out of bounds immediately and thus sacrifice state reward for fuel reward.
            # the terminating state is considered good if the lander is within the goal posts (which are at
            # x = +/-0.2) and the other orientation variables (y position, x and y velocity, angle and angular
            # velocity) are near zero. permit a small amount of lenience in the latter, since it's common for a
            # couple of the variables to be slightly positive even when the lander is sitting stationary on a flat
            # surface.
            fuel_reward = 0.0
            if abs(observation[0]) <= 0.2 and np.abs(observation[1:6]).sum() < 0.01:  # pragma no cover
                fuel_reward = self.fuel_level

            reward = state_reward + fuel_reward

        return reward


@rl_text(chapter='Feature Extractors', page=1)
class ContinuousLunarLanderFeatureExtractor(ContinuousFeatureExtractor):
    """
    Feature extractor for the continuous lunar lander environment.
    """

    def extract(
            self,
            state: GymState,
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param state: State.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature vector.
        """

        # extract raw feature values
        scaled_feature_vector = super().extract(state, refit_scaler)

        # features:
        #   0 (x pos)
        #   1 (y pos)
        #   2 (x velocity)
        #   3 (y velocity)
        #   4 (angle)
        #   5 (angular velocity)
        #   6 (leg 1 contact)
        #   7 (leg 2 contact)
        #   8 (fuel level)

        # form the one-hot state category. start by thresholding some feature values.
        state_category_feature_idxs = [0, 2, 3, 4, 5]
        state_category = OneHotCategory(*[
            value < 0.0
            for idx, value in zip(state_category_feature_idxs, state.observation[state_category_feature_idxs])
        ])

        # encode feature values
        encoded_feature_idxs = [0, 2, 3, 4, 5]
        feature_values_to_encode = scaled_feature_vector[encoded_feature_idxs]
        encoded_feature_values = self.state_category_interacter.interact(
            np.array([feature_values_to_encode]),
            [state_category]
        )[0]

        # get unencoded feature values
        both_legs_in_contact = 1.0 if np.all(state.observation[6:8] == 1.0) else 0.0
        unencoded_feature_values = np.append(scaled_feature_vector[[1, 6, 7, 8]], [both_legs_in_contact])

        # combine encoded and unencoded feature values
        final_feature_values = np.append(encoded_feature_values, unencoded_feature_values)

        return final_feature_values

    def __init__(
            self
    ):
        """
        Initialize the feature extractor.
        """

        super().__init__()

        # interact features with relevant state categories
        self.state_category_interacter = OneHotCategoricalFeatureInteracter([
            OneHotCategory(*category_args)
            for category_args in product(*([[True, False]] * 5))
        ])
