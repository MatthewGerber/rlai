from typing import Tuple, Dict

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states import State
from rlai.utils import sample_list_item


@rl_text(chapter=8, page=161)
class StochasticEnvironmentModel:
    """
    A stochastic environment model.
    """

    def update(
            self,
            state: State,
            action: Action,
            next_state: State,
            reward: Reward
    ):
        """
        Update the model with an observed subsequent state and reward, given a preceding state and action.

        :param state: State.
        :param action: Action taken in state.
        :param next_state: Subsequent state.
        :param reward: Subsequent reward.
        """

        if state not in self.state_action_next_state_reward_count:
            self.state_action_next_state_reward_count[state] = {}

        if action not in self.state_action_next_state_reward_count[state]:
            self.state_action_next_state_reward_count[state][action] = {}

        next_state_reward = (next_state, reward.r)

        if next_state_reward not in self.state_action_next_state_reward_count[state][action]:
            self.state_action_next_state_reward_count[state][action][next_state_reward] = 0

        self.state_action_next_state_reward_count[state][action][next_state_reward] += 1

    def sample(
            self,
            random_state: RandomState
    ) -> Tuple[State, Action, State, float]:
        """
        Sample the environment model.

        :param random_state: Random state.
        :return: 4-tuple of state, action, next state, and next reward.
        """

        state = sample_list_item(list(self.state_action_next_state_reward_count.keys()), None, random_state)
        action = sample_list_item(list(self.state_action_next_state_reward_count[state].keys()), None, random_state)

        next_state_reward_count = self.state_action_next_state_reward_count[state][action]
        next_state_rewards = list(next_state_reward_count.keys())
        total_count = sum(next_state_reward_count.values())
        probs = np.array([
            next_state_reward_count[next_state_reward] / total_count
            for next_state_reward in next_state_rewards
        ])

        next_state, reward = sample_list_item(next_state_rewards, probs, random_state)

        return state, action, next_state, reward

    def __init__(
            self
    ):
        """
        Initialize the environment model.
        """

        self.state_action_next_state_reward_count: Dict[State, Dict[Action, Dict[Tuple[State, float], int]]] = {}
