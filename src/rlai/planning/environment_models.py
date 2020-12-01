from abc import ABC, abstractmethod
from queue import PriorityQueue
from typing import Tuple, Dict, Optional

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states import State
from rlai.utils import sample_list_item, IncrementalSampleAverager


@rl_text(chapter=8, page=161)
class EnvironmentModel(ABC):
    """
    An environment model.
    """

    @abstractmethod
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
        pass

    @abstractmethod
    def add_state_action_priority(
            self,
            state: State,
            action: Action,
            priority: float
    ):
        """
        Add a state-action priority.

        :param state: State.
        :param action: Action.
        :param priority: Priority.
        """
        pass

    @abstractmethod
    def get_state_action_with_highest_priority(
            self,
            remove: bool
    ) -> Optional[Tuple[State, Action]]:
        """
        Get the state-action pair with the highest priority.

        :param remove: True to remove the state-action pair from the collection.
        :return: 2-tuple of state-action pair, or None if the priority queue is empty.
        """
        pass

    @abstractmethod
    def is_defined_for_state_action(
            self,
            state: State,
            action: Action
    ) -> bool:
        """
        Check whether the current model is defined for a state.

        :param state: State.
        :param action: Action.
        :return: True if defined and False otherwise.
        """
        pass


@rl_text(chapter=8, page=170)
class StochasticEnvironmentModel(EnvironmentModel):
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
    ) -> State:
        """
        Sample a previously encountered state uniformly.

        :param random_state: Random state.
        :return: State.
        """

        return sample_list_item(list(self.state_action_next_state_count.keys()), None, random_state)

    def is_defined_for_state_action(
            self,
            state: State,
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
            state: State,
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
            state: State,
            action: Action,
            random_state: RandomState
    ) -> Tuple[State, float]:
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

    def add_state_action_priority(
            self,
            state: State,
            action: Action,
            priority: float
    ):
        """
        Add a state-action priority. The addition is ignored if its priority does not exceed the threshold.

        :param state: State.
        :param action: Action.
        :param priority: Priority.
        """

        if priority > self.priority_theta:
            self.state_action_priority.put((priority, (state, action)))

    def get_state_action_with_highest_priority(
            self,
            remove: bool
    ) -> Optional[Tuple[State, Action]]:
        """
        Get the state-action pair with the highest priority.

        :param remove: True to remove the state-action pair from the collection.
        :return: 2-tuple of state-action pair, or None if the priority queue is empty.
        """

        if self.state_action_priority.empty():
            return None
        else:
            return self.state_action_priority.get()[1]

    def __init__(
            self
    ):
        """
        Initialize the environment model.
        """

        self.state_action_next_state_count: Dict[State, Dict[Action, Dict[State, int]]] = {}
        self.state_reward_averager: Dict[State, IncrementalSampleAverager] = {}
        self.priority_theta = 0.1
        self.state_action_priority: PriorityQueue = PriorityQueue()
