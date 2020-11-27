import math
import os
from argparse import ArgumentParser, Namespace
from itertools import product
from typing import List, Tuple, Any, Optional

import gym
import numpy as np
from gym.spaces import Discrete, Box
from numpy.random import RandomState

from rlai.actions import Action, DiscretizedAction
from rlai.agents.mdp import MdpAgent
from rlai.environments import Environment
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
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
            a: Action,
            agent: MdpAgent
    ) -> Tuple[State, Reward]:
        """
        Advance the state.

        :param environment: Environment.
        :param t: Time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of next state and reward.
        """

        environment: Gym

        # map discretized actions back to continuous space
        if isinstance(a, DiscretizedAction):
            gym_action = a.continuous_value
        else:
            gym_action = a.i

        observation, reward, done, _ = environment.gym_native.step(action=gym_action)

        if environment.render_current_episode:
            environment.gym_native.render()

        next_state = GymState(
            environment=environment,
            observation=observation,
            terminal=done,
            agent=agent
        )

        return next_state, Reward(i=None, r=reward)

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
            i=agent.get_state_i(observation),
            AA=environment.actions,
            terminal=terminal
        )


@rl_text(chapter='Environments', page=1)
class Gym(MdpEnvironment):
    """
    Generalized Gym environment. Any OpenAI Gym environment can be executed by supplying the appropriate identifier.
    """

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parsed_args, unparsed_args = super().parse_arguments(args)

        parser = ArgumentParser(allow_abbrev=False)

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

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

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

        parsed_args, unparsed_args = cls.parse_arguments(args)

        gym_env = Gym(
            random_state=random_state,
            **vars(parsed_args)
        )

        return gym_env, unparsed_args

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> State:
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

    def __init__(
            self,
            random_state: RandomState,
            T: Optional[int],
            gym_id: str,
            continuous_action_discretization_resolution: Optional[float] = None,
            render_every_nth_episode: Optional[int] = None,
            video_directory: Optional[str] = None
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
            raise ValueError(f'Continuous-action discretization is only valid for Box action-space environments.')

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
                self.gym_native = gym.wrappers.Monitor(self.gym_native, directory=os.path.expanduser(video_directory), video_callable=lambda episode_id: episode_id % self.render_every_nth_episode == 0)

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
