import math
import os
from argparse import ArgumentParser
from itertools import product
from typing import List, Tuple, Any, Dict, Optional

import gym
import numpy as np
from gym.spaces import Discrete, Box
from numpy.random import RandomState

from rlai.actions import Action, DiscretizedAction
from rlai.environments import Environment
from rlai.environments.mdp import MdpEnvironment
from rlai.rewards import Reward
from rlai.states import State
from rlai.states.mdp import MdpState


class GymState(MdpState):
    """
    State of a Gym environment.
    """

    def advance(
            self,
            environment: Environment,
            t: int,
            a: Action
    ) -> Tuple[State, Reward]:
        """
        Advance the state.

        :param environment: Environment.
        :param t: Time step.
        :param a: Action.
        :return: 2-tuple of next state and reward.
        """

        environment: Gym

        # map discretized actions back to continuous space
        if isinstance(a, DiscretizedAction):
            gym_action = a.continuous_value
        else:
            gym_action = a.i

        observation, reward, done, _ = environment.gym_native.step(action=gym_action)

        next_state = GymState(
            environment=environment,
            observation=observation,
            terminal=done
        )

        return next_state, Reward(i=None, r=reward)

    def __init__(
            self,
            environment,
            observation,
            terminal: bool,
    ):
        """
        Initialize the state.

        :param environment: Environment.
        :param observation: Observation.
        :param terminal: Whether the state is terminal.
        """

        environment: Gym

        super().__init__(
            i=environment.get_state_i(observation),
            AA=environment.actions,
            terminal=terminal
        )


class Gym(MdpEnvironment):
    """
    Generalized Gym environment.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Any, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parser = ArgumentParser()

        parser.add_argument(
            '--gym-id',
            type=str,
            help='Initial seed count in each pit.'
        )

        parser.add_argument(
            '--continuous-state-discretization-resolution',
            type=float,
            help='Continuous-state discretization resolution.'
        )

        parser.add_argument(
            '--continuous-action-discretization-resolution',
            type=float,
            help='Continuous-action discretization resolution.'
        )

        parser.add_argument(
            '--render-every-nth-episode',
            type=int,
            help='How often to render episodes.'
        )

        parser.add_argument(
            '--video-directory',
            type=str,
            help='Local directory in which to store rendered videos.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(args)

        gym_env = Gym(
            random_state=random_state,
            **vars(parsed_args)
        )

        return gym_env, unparsed_args

    def reset_for_new_run(
            self
    ) -> State:
        """
        Reset the environment for a new run (episode).

        :return: Initial state.
        """

        super().reset_for_new_run()

        observation = self.gym_native.reset()

        self.state = GymState(
            environment=self,
            observation=observation,
            terminal=False
        )

        return self.state

    def get_state_i(
            self,
            observation
    ) -> int:
        """
        Get the integer identifier for a state. The returned value is guaranteed to be the same for the same state,
        both throughout the life of the current object as well as after the current object has been pickled for later
        use (e.g., in checkpoint-based resumption).

        :param observation: Gym observation.
        :return: Integer identifier.
        """

        if isinstance(self.gym_native.observation_space, Box):

            if self.continuous_state_discretization_resolution is None:
                raise ValueError('Attempted to discrete a Box environment without a resolution.')

            state_id_str = '|'.join(
                str(int(state_dim_value / self.continuous_state_discretization_resolution))
                for state_dim_value in observation
            )
        else:
            raise ValueError(f'Unknown observation space type:  {type(self.gym_native.observation_space)}')

        if state_id_str not in self.state_id_str_int:
            self.state_id_str_int[state_id_str] = len(self.state_id_str_int)

        return self.state_id_str_int[state_id_str]

    def __init__(
            self,
            random_state: RandomState,
            gym_id: str,
            continuous_state_discretization_resolution: Optional[float] = None,
            continuous_action_discretization_resolution: Optional[float] = None,
            render_every_nth_episode: Optional[int] = None,
            video_directory: Optional[str] = None
    ):
        """
        Initialize the environment.

        :param random_state: Random state.
        :param gym_id: Gym identifier. See https://gym.openai.com/envs for a list.
        :param continuous_state_discretization_resolution: A discretization resolution for continuous-state
        environments. Providing this value allows the environment to be used with discrete-state methods via
        discretization of the continuous-state dimensions.
        :param continuous_action_discretization_resolution: A discretization resolution for continuous-action
        environments. Providing this value allows the environment to be used with discrete-action methods via
        discretization of the continuous-action dimensions.
        :param render_every_nth_episode: If passed, the environment will render an episode video per this value.
        :param video_directory: Directory in which to store rendered videos.
        """

        super().__init__(
            name=f'gym ({gym_id})',
            random_state=random_state
        )

        self.gym_native = gym.make(
            id=gym_id
        )

        if continuous_state_discretization_resolution is not None and not isinstance(self.gym_native.observation_space, Box):
            raise ValueError(f'Continuous-state discretization is only valid for Box state-space environments.')

        if continuous_action_discretization_resolution is not None and not isinstance(self.gym_native.action_space, Box):
            raise ValueError(f'Continuous-action discretization is only valid for Box action-space environments.')

        if render_every_nth_episode is not None:
            self.gym_native = gym.wrappers.Monitor(self.gym_native, directory=os.path.expanduser(video_directory), video_callable=lambda episode_id: episode_id % render_every_nth_episode == 0, force=True)

        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution
        self.gym_native.seed(random_state.randint(1000))
        self.state_id_str_int: Dict[str, int] = {}

        # action space is already discrete. initialize n actions.
        if isinstance(self.gym_native.action_space, Discrete):
            self.actions = [
                Action(
                    i=i
                )
                for i in range(self.gym_native.action_space.n)
            ]

        # action space is continuous. discretize.
        elif isinstance(self.gym_native.action_space, Box):

            box = self.gym_native.action_space

            # continuous n-dimensional action space with identical bounds on each dimension
            if len(box.shape) == 1:
                action_discretizations = [
                    np.linspace(low, high, math.ceil((high - low) / continuous_action_discretization_resolution))
                    for low, high in zip(box.low, box.high)
                ]
            else:
                raise ValueError(f'Unknown format of continuous action space:  {box}')

            self.actions = [
                DiscretizedAction(
                    i=i,
                    continuous_value=np.array(n_dim_action)
                )
                for i, n_dim_action in enumerate(product(*action_discretizations))
            ]

        else:
            raise ValueError(f'Unknown Gym action space type:  {type(self.gym_native.action_space)}')
