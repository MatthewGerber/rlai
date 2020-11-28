from typing import Tuple, Dict

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.rewards import Reward
from rlai.states import State
from rlai.utils import sample_list_item


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
        Update the model with an observed next state and reward, given a preceding state and action.

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
            state: State,
            action: Action
    ) -> Tuple[State, float]:
        """
        Sample a subsequent state and reward given a current state and action.

        :param state: Current state.
        :param action: Action.
        :return: 2-tuple of subsequent state and reward.
        """

        next_state_reward_count = self.state_action_next_state_reward_count[state][action]
        next_state_rewards = list(next_state_reward_count.keys())
        total_count = sum(next_state_reward_count.values())
        probs = np.array([
            next_state_reward_count[next_state_reward] / total_count
            for next_state_reward in next_state_rewards
        ])

        return sample_list_item(next_state_rewards, probs, self.random_state)

    def __init__(
            self,
            random_state: RandomState
    ):
        """
        Initialize the environment model.

        :param random_state: Random State
        """

        self.random_state = random_state

        self.state_action_next_state_reward_count: Dict[State, Dict[Action, Dict[Tuple[State, float], int]]] = {}
