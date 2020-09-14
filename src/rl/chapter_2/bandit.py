import sys
from typing import List

from numpy.random import RandomState
from scipy.stats import norm
import matplotlib.pyplot as plt
from rl.utils import IncrementalAverager


class Agent:

    def reset(
            self
    ):
        self.Q = {
            a: IncrementalAverager()
            for a in self.AA
        }
        self.best_A = list(self.Q.keys())[0]

    def act(
            self
    ) -> int:

        if self.act_random.random_sample() < self.epsilon:
            A = self.act_random.choice(self.AA)
            self.epsilon *= (1 - self.epsilon_reduction_rate)
        else:
            A = self.best_A

        return A

    def reward(
            self,
            action: int,
            reward: float
    ):
        self.Q[action].update(reward)
        self.best_A = max(self.Q.items(), key=lambda action_value: action_value[1].value())[0]

    def __init__(
            self,
            AA: List[int],
            epsilon: float,
            epsilon_reduction_rate: float,
            seed: int
    ):
        self.AA = AA
        self.epsilon = epsilon
        self.epsilon_reduction_rate = epsilon_reduction_rate
        self.seed = seed

        self.Q = {}
        self.best_A = None
        self.act_random = RandomState(seed)
        self.reset()


class Arm:

    def reset(self):
        self.random_state = RandomState(self.seed)

    def pull(
            self
    ) -> float:

        return self.q_star.rvs(1, random_state=self.random_state)[0]

    def __init__(
            self,
            i: int,
            mean: float,
            variance: float,
            seed: int = None
    ):
        self.i = i
        self.mean = mean
        self.variance = variance
        self.seed = seed

        self.q_star = norm(loc=mean, scale=variance)
        self.random_state = RandomState(self.seed)
        self.reset()

    def __str__(
            self
    ) -> str:
        return f'Mean:  {self.mean}, Variance:  {self.variance}'


class KArmedBandit:

    def reset(
            self
    ):
        self.arms = [

            Arm(
                i=i,
                mean=self.q_star_random.normal(loc=self.q_star_mean, scale=self.q_star_variance, size=1)[0],
                variance=self.reward_variance,
                seed=self.seed
            )

            for i in range(self.k)
        ]

        self.best_arm = max(self.arms, key=lambda arm: arm.mean).i

    def pull(
            self,
            arm: int
    ) -> float:

        return self.arms[arm].pull()

    def run(
            self,
            agent: Agent,
            T: int,
            n_runs: int
    ):
        t_average_reward = [
            IncrementalAverager()
            for _ in range(T)
        ]

        reset_random = RandomState(self.seed)

        for i in range(n_runs):

            if (i % 100) == 0:
                print(f'Starting run {i + 1} of {n_runs}...')

            for t in range(T):

                if reset_random.random_sample() < self.reset_probability:
                    self.reset()

                action = agent.act()
                reward = self.pull(action)
                agent.reward(action, reward)

                t_average_reward[t].update(reward)

            self.reset()
            agent.reset()

        return t_average_reward

    def __init__(
            self,
            k: int,
            q_star_mean: float,
            q_star_variance: float,
            reward_variance: float,
            reset_probability: float,
            seed: int
    ):
        self.k = k
        self.q_star_mean = q_star_mean
        self.q_star_variance = q_star_variance
        self.reward_variance = reward_variance
        self.reset_probability = reset_probability
        self.seed = seed

        self.q_star_random = RandomState(seed)
        self.arms = []
        self.best_arm = None
        self.reset()


def main(
        argv
):
    k = 10

    bandit = KArmedBandit(
        k=k,
        q_star_mean=0,
        q_star_variance=1,
        reward_variance=1,
        reset_probability=0,
        seed=203948
    )

    agent = Agent(
        AA=list(range(k)),
        epsilon=0.1,
        epsilon_reduction_rate=0,
        seed=234234
    )

    t_average_reward = bandit.run(
        agent=agent,
        T=1000,
        n_runs=2000
    )

    plt.plot([averager.value() for averager in t_average_reward])
    plt.show()


if __name__ == '__main__':
    main(sys.argv[1:])
