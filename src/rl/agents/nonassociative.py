from typing import List, Dict, Optional

from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.environments.state import State
from rl.meta import rl_text
from rl.utils import IncrementalSampleAverager


@rl_text(chapter=2, page=27)
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

        self.epsilon = self.original_epsilon

        for averager in self.Q.values():
            averager.reset()

        self.greedy_action = list(self.Q.keys())[0]
        self.most_recent_action = None

    def sense(
            self,
            state: State
    ):
        """
        No effect (the agent is nonassociative).

        :param state: State.
        """
        pass

    def act(
            self
    ) -> Action:
        """
        Act in an epsilon-greedy fashion.

        :return: Action.
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

        # get new greedy action, which might have changed
        self.greedy_action = max(self.Q.items(), key=lambda action_value: action_value[1].get_value())[0]

    def __init__(
            self,
            AA: List[Action],
            name: str,
            epsilon: float,
            epsilon_reduction_rate: float,
            alpha: float,
            random_state: RandomState
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of agent.
        :param epsilon: Epsilon.
        :param epsilon_reduction_rate: Rate at which `epsilon` is reduced from its initial value to zero. This is the
        percentage reduction, and it is applied at each time step when the agent explores. For example, pass 0 for no
        reduction and 0.01 for a 1-percent reduction at each exploration step.
        :param alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        :param random_state: Random state.
        """

        super().__init__(
            AA=AA,
            name=name
        )

        self.epsilon = self.original_epsilon = epsilon
        self.epsilon_reduction_rate = epsilon_reduction_rate
        self.random_state = random_state

        self.Q: Dict[Action, IncrementalSampleAverager] = {
            a: IncrementalSampleAverager(
                alpha=alpha
            )
            for a in self.AA
        }

        self.greedy_action: Optional[Action] = None
        self.most_recent_action: Optional[Action] = None
        self.reset()
