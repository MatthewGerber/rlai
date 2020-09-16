from typing import Dict

from rl.agents.action import Action
from rl.utils import IncrementalSampleAverager


class Monitor:
    """
    Monitor for runs of an environment with an agent.
    """

    def reset(
            self
    ):
        """
        Reset the monitor.
        """

        for averager in self.t_average_reward:
            averager.reset()

        for t in range(self.T):
            self.t_count_optimal_action[t] = 0

    def report(
            self,
            t: int,
            agent_action: Action = None,
            optimal_action: Action = None,
            action_reward: float = None
    ):
        """
        Report information about a run.

        :param t: Time step.
        :param agent_action: Action taken.
        :param optimal_action: Optimal action.
        :param action_reward: Reward obtained.
        """

        if agent_action is not None and optimal_action is not None and agent_action == optimal_action:
            self.t_count_optimal_action[t] += 1

        if action_reward is not None:
            self.t_average_reward[t].update(action_reward)

    def __init__(
            self,
            T: int
    ):
        """
        Initialize the monitor.

        :param T: Number of time steps in run.
        """

        self.T = T

        self.t_average_reward = [
            IncrementalSampleAverager()
            for _ in range(self.T)
        ]

        self.t_count_optimal_action = [0] * self.T
