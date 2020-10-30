from argparse import Namespace, ArgumentParser
from typing import List, Optional, Tuple

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.environments import Environment
from rlai.meta import rl_text
from rlai.runners.monitor import Monitor
from rlai.states import State

ARM_QSTAR_BUFFER_SIZE = 1000


@rl_text(chapter=2, page=25)
class Arm:
    """
    Bandit arm.
    """

    def pull(
            self
    ) -> float:
        """
        Pull the arm.

        :return: Reward value.
        """

        # refill the reward buffer if it is empty or hasn't been initialized
        if self.q_star_buffer_idx >= len(self.q_star_buffer):
            self.q_star_buffer = self.random_state.normal(loc=self.mean, scale=self.variance, size=ARM_QSTAR_BUFFER_SIZE)
            self.q_star_buffer_idx = 0

        # return next value from buffer
        value = self.q_star_buffer[self.q_star_buffer_idx]
        self.q_star_buffer_idx += 1

        return value

    def __init__(
            self,
            i: int,
            mean: float,
            variance: float,
            random_state: RandomState
    ):
        """
        Initialize the arm.

        :param i: Arm index.
        :param mean: Mean reward value.
        :param variance: Variance of reward value.
        :param random_state: Random state.
        """

        self.i: int = i
        self.mean: float = mean
        self.variance: float = variance
        self.random_state: RandomState = random_state

        self.q_star_buffer: np.ndarray = np.array([])
        self.q_star_buffer_idx: int = 0

    def __str__(
            self
    ) -> str:
        return f'Mean:  {self.mean}, Variance:  {self.variance}'


@rl_text(chapter=2, page=28)
class KArmedBandit(Environment):
    """
    K-armed bandit.
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
            '--k',
            type=int,
            default=10,
            help='Number of bandit arms.'
        )

        parser.add_argument(
            '--reset-probability',
            type=float,
            default=0.0,
            help="Probability of resetting the bandit's arms at each time step. This effectively creates a nonstationary environment."
        )

        parser.add_argument(
            '--q-star-mean',
            type=float,
            default=0.0,
            help='Mean of q-star (true reward mean) distribution.'
        )

        parser.add_argument(
            '--q-star-variance',
            type=float,
            default=1.0,
            help='Variance of q-star (true reward mean) distribution.'
        )

        parser.add_argument(
            '--reward-variance',
            type=float,
            default=1.0,
            help='Variance of rewards.'
        )

        return parser.parse_known_args(unparsed_args, parsed_args)

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

        parsed_args, unparsed_args = cls.parse_arguments(args)

        bandit = KArmedBandit(
            random_state=random_state,
            **dict(parsed_args._get_kwargs())
        )

        return bandit, unparsed_args

    def reset_for_new_run(
            self
    ) -> State:
        """
        Reset the the bandit, initializing arms to new expected values.

        :return: New State.
        """

        # get new arm reward means and initialize new arms
        q_star_means = self.random_state.normal(loc=self.q_star_mean, scale=self.q_star_variance, size=self.k)

        self.arms = [
            Arm(
                i=i,
                mean=mean,
                variance=self.reward_variance,
                random_state=self.random_state
            )
            for i, mean in enumerate(q_star_means)
        ]

        self.best_arm = max(self.arms, key=lambda arm: arm.mean)

        return State(i=0, AA=[Action(i) for i in range(self.k)])

    def pull(
            self,
            arm: int
    ) -> float:
        """
        Pull an arm.

        :param arm: Arm index.
        :return: Reward value.
        """

        return self.arms[arm].pull()

    def run_step(
            self,
            t: int,
            agent: Agent,
            monitor: Monitor
    ) -> bool:
        """
        Run a step of the environment with an agent.

        :param t: Step.
        :param agent: Agent.
        :param monitor: Monitor.
        :return: True if a terminal state was entered and the run should terminate, and False otherwise.
        """

        if self.random_state.random_sample() < self.reset_probability:
            self.reset_for_new_run()

        action = agent.act(t=t)
        monitor.report(t=t, agent_action=action, optimal_action=Action(self.best_arm.i))

        reward = self.pull(action.i)
        monitor.report(t=t, action_reward=reward)

        agent.reward(reward)

        return False

    def __init__(
            self,
            random_state: RandomState,
            k: int,
            q_star_mean: float,
            q_star_variance: float,
            reward_variance: float,
            reset_probability: float
    ):
        """
        Initialize the bandit.

        :param k: Number of arms.
        :param q_star_mean: Mean of q_star.
        :param q_star_variance: Variance of q_star.
        :param reward_variance: Reward variance.
        :param reset_probability: Per-step probability of resetting (nonstationarity).
        :param random_state: Random state.
        """

        super().__init__(
            name=f'{k}-armed bandit',
            random_state=random_state
        )

        self.k = k
        self.q_star_mean = q_star_mean
        self.q_star_variance = q_star_variance
        self.reward_variance = reward_variance
        self.reset_probability = reset_probability

        self.arms: List[Arm] = []
        self.best_arm: Optional[Arm] = None
