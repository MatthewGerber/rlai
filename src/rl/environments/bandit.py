import sys

import matplotlib.pyplot as plt
from numpy.random import RandomState
from scipy import stats

from rl.agents.base import Agent
from rl.agents.nonassociative import EpsilonGreedy
from rl.utils import OnlineSampleAverager


class Arm:
    """
    Bandit arm.
    """

    def reset(
            self
    ):
        """
        Reset the arm.
        """

        self.q_star_buffer = []
        self.q_star_buffer_idx = None

    def pull(
            self
    ) -> float:
        """
        Pull the arm.

        :return: Reward value.
        """

        # refill the reward buffer if it is empty or hasn't been initialized
        if len(self.q_star_buffer) == 0 or self.q_star_buffer_idx is None or self.q_star_buffer_idx >= len(self.q_star_buffer):
            self.q_star_buffer = list(self.q_star.rvs(1000, random_state=self.random_state))
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

        self.i = i
        self.mean = mean
        self.variance = variance
        self.random_state = random_state

        self.q_star = stats.norm(loc=mean, scale=variance)
        self.q_star_buffer = []
        self.q_star_buffer_idx = None
        self.reset()

    def __str__(
            self
    ) -> str:
        return f'Mean:  {self.mean}, Variance:  {self.variance}'


class KArmedBandit:
    """
    K-armed bandit.
    """

    def reset_arms(
            self
    ):
        """
        Reset the arms of the bandit, initializing them to new expected values.
        """

        # get new arm reward means and initialize new arms
        q_star_means = self.random_state.normal(loc=self.q_star_mean, scale=self.q_star_variance, size=self.k)

        self.arms = [

            Arm(
                i=i,
                mean=q_star_means[i],
                variance=self.reward_variance,
                random_state=self.random_state
            )

            for i, mean in enumerate(q_star_means)
        ]

        self.best_arm = max(self.arms, key=lambda arm: arm.mean).i

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

    def run(
            self,
            agent: Agent,
            T: int,
            n_runs: int
    ):
        t_average_reward = [
            OnlineSampleAverager()
            for _ in range(T)
        ]

        for i in range(n_runs):

            for t in range(T):

                if self.random_state.random_sample() < self.reset_probability:
                    self.reset_arms()

                action = agent.act()
                reward = self.pull(action)
                agent.reward(reward)

                t_average_reward[t].update(reward)

            runs_finished = i + 1
            if (runs_finished % 100) == 0:
                percent_done = 100 * (runs_finished / n_runs)
                print(f'{percent_done:.0f}% complete ({runs_finished} of {n_runs})...')

            self.reset_arms()
            agent.reset()

        return t_average_reward

    def __init__(
            self,
            k: int,
            q_star_mean: float,
            q_star_variance: float,
            reward_variance: float,
            reset_probability: float,
            random_state: RandomState
    ):
        self.k = k
        self.q_star_mean = q_star_mean
        self.q_star_variance = q_star_variance
        self.reward_variance = reward_variance
        self.reset_probability = reset_probability
        self.random_state = random_state

        self.arms = []
        self.best_arm = None
        self.reset_arms()
