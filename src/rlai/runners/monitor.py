from typing import Optional

from rlai.actions import Action
from rlai.utils import IncrementalSampleAverager


class Monitor:
    """
    Monitor for runs of an environment with an agent.
    """

    def reset_for_new_run(
            self
    ):
        """
        Reset the monitor for a new run.
        """

        self.cumulative_reward = 0.0

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

        if t not in self.t_count_optimal_action:
            self.t_count_optimal_action[t] = 0

        if t not in self.t_average_reward:
            self.t_average_reward[t] = IncrementalSampleAverager()

        if t not in self.t_average_cumulative_reward:
            self.t_average_cumulative_reward[t] = IncrementalSampleAverager()

        if agent_action is not None and optimal_action is not None and agent_action == optimal_action:
            self.t_count_optimal_action[t] += 1

        if action_reward is not None:
            self.t_average_reward[t].update(action_reward)
            self.cumulative_reward += action_reward
            self.t_average_cumulative_reward[t].update(self.cumulative_reward)

        self.most_recent_time_step = t

    def __init__(
            self
    ):
        """
        Initialize the monitor.
        """

        self.t_count_optimal_action = {}
        self.t_average_reward = {}
        self.t_average_cumulative_reward = {}
        self.cumulative_reward = 0.0
        self.most_recent_time_step: Optional[int] = None
