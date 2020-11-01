from abc import abstractmethod, ABC
from typing import List, Dict, Tuple, Optional

import numpy as np

from rlai.actions import Action
from rlai.environments import Environment
from rlai.meta import rl_text
from rlai.rewards import Reward
from rlai.states import State
from rlai.utils import sample_list_item


@rl_text(chapter=3, page=47)
class MdpState(State, ABC):
    """
    Model-free MDP state.
    """

    @abstractmethod
    def advance(
            self,
            environment: Environment,
            t: int,
            a: Action
    ) -> Tuple[State, int, Reward]:
        """
        Advance from the current state given an action.

        :param environment: Environment.
        :param t: Current time step.
        :param a: Action.
        :return: 3-tuple of next state, next time step, and next reward.
        """
        pass

    def __init__(
            self,
            i: Optional[int],
            AA: List[Action],
            terminal: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions that can be taken from this state.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=i,
            AA=AA
        )

        self.terminal = terminal


@rl_text(chapter=3, page=47)
class ModelBasedMdpState(MdpState):
    """
    Model-based MDP state. Adds the specification of a probability distribution over next states and rewards.
    """

    def initialize_model(
            self,
            SS: List[State],
            RR: List[Reward]
    ):
        """
        Initialize the model within the current state to all zeros. This does not result in a valid model; rather it is
        a starting point for further specification.

        :param SS: All states that can follow the current one based on the current state's actions.
        :param RR: All rewards that can be obtained from the following states `SS`.
        """

        self.p_S_prime_R_given_A = {
            a: {
                s: {
                    r: 0.0
                    for r in RR
                }
                for s in SS
            }
            for a in self.AA
        }

    def check_marginal_probabilities(
            self
    ):
        """
        Check the marginal next-state and next-reward probabilities, to ensure they sum to 1. Raises an exception if
        this is not the case.
        """

        for a in self.p_S_prime_R_given_A:

            marginal_prob = sum([
                self.p_S_prime_R_given_A[a][s_prime][r]
                for s_prime in self.p_S_prime_R_given_A[a]
                for r in self.p_S_prime_R_given_A[a][s_prime]
            ])

            if marginal_prob != 1.0:
                raise ValueError(f'Expected next-state/next-reward marginal probability of 1.0, but got {marginal_prob}.')

    def advance(
            self,
            environment: Environment,
            t: int,
            a: Action
    ) -> Tuple[State, int, Reward]:
        """
        Advance from the current state given an action, based on the current state's model probability distribution.

        :param environment: Environment.
        :param t: Current time step.
        :param a: Action.
        :return: 3-tuple of next state, next time step, and next reward.
        """

        # get next-state / reward tuples
        s_prime_rewards = [
            (s_prime, reward)
            for s_prime in self.p_S_prime_R_given_A[a]
            for reward in self.p_S_prime_R_given_A[a][s_prime]
            if self.p_S_prime_R_given_A[a][s_prime][reward] > 0.0
        ]

        # get probability of each tuple
        probs = np.array([
            self.p_S_prime_R_given_A[a][s_prime][reward]
            for s_prime in self.p_S_prime_R_given_A[a]
            for reward in self.p_S_prime_R_given_A[a][s_prime]
            if self.p_S_prime_R_given_A[a][s_prime][reward] > 0.0
        ])

        # sample next state and reward
        next_state, next_reward = sample_list_item(
            x=s_prime_rewards,
            probs=probs,
            random_state=environment.random_state
        )

        return next_state, t + 1, next_reward

    def __init__(
            self,
            i: int,
            AA: List[Action],
            terminal: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions that can be taken from this state.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=i,
            AA=AA,
            terminal=terminal
        )

        # initialize an empty model within the state (see `initialize_model` for initialization)
        self.p_S_prime_R_given_A: Dict[
            Action, Dict[
                ModelBasedMdpState, Dict[
                    Reward, float
                ]
            ]
        ] = {}
