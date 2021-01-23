from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states.mdp import MdpState
from rlai.utils import sample_list_item, IncrementalSampleAverager


@rl_text(chapter=8, page=161)
class EnvironmentModel(ABC):
    """
    An environment model.
    """

    @abstractmethod
    def update(
            self,
            state: MdpState,
            action: Action,
            next_state: MdpState,
            reward: Reward
    ):
        """
        Update the model with an observed subsequent state and reward, given a preceding state and action.

        :param state: State.
        :param action: Action taken in state.
        :param next_state: Subsequent state.
        :param reward: Subsequent reward.
        """

    @abstractmethod
    def is_defined_for_state_action(
            self,
            state: MdpState,
            action: Action
    ) -> bool:
        """
        Check whether the current model is defined for a state.

        :param state: State.
        :param action: Action.
        :return: True if defined and False otherwise.
        """


@rl_text(chapter=8, page=170)
class StochasticEnvironmentModel(EnvironmentModel):
    """
    A stochastic environment model.
    """

    def update(
            self,
            state: MdpState,
            action: Action,
            next_state: MdpState,
            reward: Reward
    ):
        """
        Update the model with an observed subsequent state and reward, given a preceding state and action.

        :param state: State.
        :param action: Action taken in state.
        :param next_state: Subsequent state.
        :param reward: Subsequent reward.
        """

        if state not in self.state_action_next_state_count:
            self.state_action_next_state_count[state] = {}

        if action not in self.state_action_next_state_count[state]:
            self.state_action_next_state_count[state][action] = {}

        if next_state not in self.state_action_next_state_count[state][action]:
            self.state_action_next_state_count[state][action][next_state] = 0

        self.state_action_next_state_count[state][action][next_state] += 1

        if next_state not in self.state_reward_averager:
            self.state_reward_averager[next_state] = IncrementalSampleAverager()

        self.state_reward_averager[next_state].update(reward.r)

    def sample_state(
            self,
            random_state: RandomState
    ) -> MdpState:
        """
        Sample a previously encountered state uniformly.

        :param random_state: Random state.
        :return: State.
        """

        return sample_list_item(list(self.state_action_next_state_count.keys()), None, random_state)

    def is_defined_for_state_action(
            self,
            state: MdpState,
            action: Action
    ) -> bool:
        """
        Check whether the current model is defined for a state-action pair.

        :param state: State.
        :param action: Action.
        :return: True if defined and False otherwise.
        """

        return state in self.state_action_next_state_count and action in self.state_action_next_state_count[state]

    def sample_action(
            self,
            state: MdpState,
            random_state: RandomState
    ) -> Action:
        """
        Sample a previously encountered action in a given state uniformly.

        :param state: State.
        :param random_state: Random state.
        :return: Action
        """

        return sample_list_item(list(self.state_action_next_state_count[state].keys()), None, random_state)

    def sample_next_state_and_reward(
            self,
            state: MdpState,
            action: Action,
            random_state: RandomState
    ) -> Tuple[MdpState, float]:
        """
        Sample the environment model.

        :param state: State.
        :param action: Action.
        :param random_state: Random state.
        :return: 2-tuple of next state and reward.
        """

        # sample next state
        next_state_count = self.state_action_next_state_count[state][action]
        next_states = list(next_state_count.keys())
        total_count = sum(next_state_count.values())

        probs = np.array([
            next_state_count[next_state] / total_count
            for next_state in next_states
        ])

        next_state = sample_list_item(next_states, probs, random_state)

        # get average reward in next state
        reward = self.state_reward_averager[next_state].get_value()

        return next_state, reward

    def __init__(
            self
    ):
        """
        Initialize the environment model.
        """

        self.state_action_next_state_count: Dict[MdpState, Dict[Action, Dict[MdpState, int]]] = {}
        self.state_reward_averager: Dict[MdpState, IncrementalSampleAverager] = {}
