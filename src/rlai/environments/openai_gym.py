from typing import List, Tuple, Any, Dict, Optional

import gym
from gym.spaces import Discrete, Box
from numpy.random import RandomState

from rlai.actions import Action
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

        observation, reward, done, _ = environment.gym_native.step(action=a.i)

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
        pass

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
            continuous_state_discretization_resolution: Optional[float] = None
    ):
        """
        Initialize the environment.

        :param random_state: Random state.
        :param gym_id: Gym identifier. See https://gym.openai.com/envs for a list.
        :param continuous_state_discretization_resolution: A discretization resolution for continuous-state
        environments. Providing this value allows the environment to be used with discrete-state methods via
        discretization of the continuous-state dimensions.
        """

        self.gym_native = gym.make(
            id=gym_id
        )

        self.gym_native.seed(random_state.randint(1000))

        if continuous_state_discretization_resolution is not None and not isinstance(self.gym_native.observation_space, Box):
            raise ValueError(f'Continuous-state discretization is only valid for Box environments.')

        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution
        self.state_id_str_int: Dict[str, int] = {}

        if isinstance(self.gym_native.action_space, Discrete):
            self.actions = [
                Action(
                    i=i
                )
                for i in range(self.gym_native.action_space.n)
            ]
        else:
            raise ValueError(f'Unknown Gym action space type:  {type(self.gym_native.action_space)}')

        super().__init__(
            name=f'gym ({gym_id})',
            random_state=random_state
        )
