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

    def report(
            self,
            t: int,
            agent_action: Action = None,
            action_reward: float = None
    ):
        """
        Report information about a run.

        :param t: Time step.
        :param agent_action: Action taken.
        :param action_reward: Reward obtained.
        """

        if action_reward is not None:
            self.t_average_reward[t].update(action_reward)

    def __init__(
            self,
            T: int
    ):
        """
        Initialize the monitor.

        :param T: Number of time steps in the run.
        """

        self.t_average_reward = [
            IncrementalSampleAverager()
            for _ in range(T)
        ]
