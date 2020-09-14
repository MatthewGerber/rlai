import sys
from typing import List

import matplotlib.pyplot as plt
from numpy.random import RandomState
from scipy import stats

from rl.utils import OnlineSampleAverager


class Agent:
    """
    An agent for playing a `KArmedBandit`.
    """

    def reset_action_value_function(
            self
    ):
        """
        Reset the action-value funtion.
        """

        self.Q = {
            a: OnlineSampleAverager()
            for a in self.AA
        }

        self.greedy_action = list(self.Q.keys())[0]

    def act(
            self
    ) -> int:
        """
        Act in an epsilon-greedy fashion.

        :return: Action number.
        """

        if self.random_state.random_sample() < self.epsilon:
            a = self.random_state.choice(self.AA)
            self.epsilon *= (1 - self.epsilon_reduction_rate)
        else:
            a = self.greedy_action

        return a

    def reward(
            self,
            a: int,
            r: float
    ):
        """
        Reward the current `Agent`.

        :param a: Action that produced the reward.
        :param r: Reward value.
        """

        self.Q[a].update(r)
        self.greedy_action = max(self.Q.items(), key=lambda action_value: action_value[1].get_value())[0]

    def __init__(
            self,
            AA: List[int],
            epsilon: float,
            epsilon_reduction_rate: float,
            random_state: RandomState
    ):
        """
        Initialize the `Agent`.

        :param AA: Set of all possible actions.
        :param epsilon: Epsilon.
        :param epsilon_reduction_rate: Epsilon reduction rate (per time tick).
        :param random_state: Random state.
        """

        self.AA = AA
        self.epsilon = epsilon
        self.epsilon_reduction_rate = epsilon_reduction_rate
        self.random_state = random_state

        self.Q = {}
        self.greedy_action = None
        self.reset_action_value_function()


class Arm:
    """
    Arm of a bandit.
    """

    def reset(
            self
    ):
        """
        Reset the `Arm`.
        """

        self.q_star_buffer = []
        self.q_star_buffer_idx = None

    def pull(
            self
    ) -> float:
        """
        Pull the `Arm`.
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
        Initialize the `Arm`.

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
        Reset the arms of the `KArmedBandit`.
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

            if (i % 100) == 0:
                print(f'Starting run {i + 1} of {n_runs}...')

            for t in range(T):

                if self.random_state.random_sample() < self.reset_probability:
                    self.reset_arms()

                action = agent.act()
                reward = self.pull(action)
                agent.reward(action, reward)

                t_average_reward[t].update(reward)

            self.reset_arms()
            agent.reset_action_value_function()

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


def main(
        argv
):
    k = 10

    random_state = RandomState(12345)

    bandit = KArmedBandit(
        k=k,
        q_star_mean=0,
        q_star_variance=1,
        reward_variance=1,
        reset_probability=0,
        random_state=random_state
    )

    agent = Agent(
        AA=list(range(k)),
        epsilon=0.1,
        epsilon_reduction_rate=0,
        random_state=random_state
    )

    t_average_reward = bandit.run(
        agent=agent,
        T=1000,
        n_runs=2000
    )

    plt.plot([averager.get_value() for averager in t_average_reward])
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
