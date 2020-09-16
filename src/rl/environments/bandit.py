from typing import List, Optional

from numpy.random import RandomState
from scipy import stats

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.environments.base import Environment
from rl.meta import rl_text
from rl.runners.monitor import Monitor


@rl_text(chapter=2, page=25)
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

        self.q_star_buffer.clear()
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

        self.i: int = i
        self.mean: float = mean
        self.variance: float = variance
        self.random_state: RandomState = random_state

        self.q_star = stats.norm(loc=mean, scale=variance)
        self.q_star_buffer: List[float] = []
        self.q_star_buffer_idx: Optional[int] = None
        self.reset()

    def __str__(
            self
    ) -> str:
        return f'Mean:  {self.mean}, Variance:  {self.variance}'


@rl_text(chapter=2, page=28)
class KArmedBandit(Environment):
    """
    K-armed bandit.
    """

    def reset(
            self
    ):
        """
        Reset the the bandit, initializing arms to new expected values.
        """

        # get new arm reward means and initialize new arms
        q_star_means = self.random_state.normal(loc=self.q_star_mean, scale=self.q_star_variance, size=self.k)

        self.arms.clear()
        self.arms.extend([

            Arm(
                i=i,
                mean=q_star_means[i],
                variance=self.reward_variance,
                random_state=self.random_state
            )

            for i, mean in enumerate(q_star_means)
        ])

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
            monitor: Monitor
    ):
        """
        Run the environment with an agent.

        :param agent: Agent to run.
        :param T: Number of time steps to run.
        :param monitor: Monitor.
        """

        for t in range(T):

            if self.random_state.random_sample() < self.reset_probability:
                self.reset()

            action = agent.act()
            monitor.report(t=t, agent_action=action, optimal_action=Action(self.best_arm))

            reward = self.pull(action.i)
            monitor.report(t=t, action_reward=reward)

            agent.reward(reward)

    def __init__(
            self,
            name: str,
            k: int,
            q_star_mean: float,
            q_star_variance: float,
            reward_variance: float,
            reset_probability: float,
            random_state: RandomState
    ):
        """
        Initialize the bandit.

        :param name: Name of the environment.
        :param k: Number of arms.
        :param q_star_mean: Mean of q_star.
        :param q_star_variance: Variance of q_star.
        :param reward_variance: Reward variance.
        :param reset_probability: Per-step probability of resetting (nonstationarity).
        :param random_state: Random state.
        """

        super().__init__(
            name=name
        )

        self.k = k
        self.q_star_mean = q_star_mean
        self.q_star_variance = q_star_variance
        self.reward_variance = reward_variance
        self.reset_probability = reset_probability
        self.random_state = random_state

        self.arms: List[Arm] = []
        self.best_arm: Optional[Arm] = None
        self.reset()
