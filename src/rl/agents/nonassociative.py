from typing import List

from numpy.random.mtrand import RandomState

from rl.agents.base import Agent
from rl.environments.state import State
from rl.meta import rl_text
from rl.utils import OnlineSampleAverager


@rl_text(page=20)
class EpsilonGreedy(Agent):
    """
    An epsilon-greedy agent.
    """

    def reset(
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

    def sense(
            self,
            state: State
    ):
        pass

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

        self.most_recent_action = a

        return a

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward value.
        """

        if self.most_recent_action is not None:
            self.Q[self.most_recent_action].update(r)

        self.greedy_action = max(self.Q.items(), key=lambda action_value: action_value[1].get_value())[0]

    def __init__(
            self,
            AA: List[int],
            epsilon: float,
            epsilon_reduction_rate: float,
            random_state: RandomState
    ):
        """
        Initialize the agent.

        :param AA: Set of all possible actions.
        :param epsilon: Epsilon.
        :param epsilon_reduction_rate: Epsilon reduction rate (per time tick).
        :param random_state: Random state.
        """

        super().__init__(
            AA=AA
        )

        self.epsilon = epsilon
        self.epsilon_reduction_rate = epsilon_reduction_rate
        self.random_state = random_state

        self.Q = {}
        self.greedy_action = None
        self.most_recent_action = None
        self.reset()
