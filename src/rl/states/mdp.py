from abc import abstractmethod, ABC
from typing import List, Dict, Tuple

import numpy as np
from numpy.random import RandomState

from rl.actions import Action
from rl.meta import rl_text
from rl.rewards import Reward
from rl.states import State
from rl.utils import sample_list_item


@rl_text(chapter=3, page=47)
class MdpState(State, ABC):
    """
    Model-free MDP state.
    """

    def is_feasible(
            self,
            a: Action
    ) -> bool:
        """
        Check whether an action is feasible from the current state. This uses a set-based lookup with O(1) complexity,
        which is far faster than checking for the action in self.AA.

        :param a: Action.
        :return: True if the action is feasible from the current state and False otherwise.
        """

        return a in self.AA_set

    @abstractmethod
    def advance(
            self,
            a: Action,
            t: int,
            random_state: RandomState
    ) -> Tuple[State, Reward]:
        """
        Advance from the current state given an action.

        :param a: Action.
        :param t: Current time step.
        :param random_state: Random state.
        :return: 2-tuple of next state and reward.
        """
        pass

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
            i=i
        )

        self.AA = AA
        self.terminal = terminal

        # use set for fast existence checks (e.g., in `feasible` function)
        self.AA_set = set(self.AA)


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
            a: Action,
            t: int,
            random_state: RandomState
    ) -> Tuple[State, Reward]:
        """
        Advance from the current state given an action, based on the current state's model probability distribution.

        :param a: Action.
        :param t: Current time step.
        :param random_state: Random state.
        :return: 2-tuple of next state and reward.
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

        # sample next-state / reward
        return sample_list_item(
            x=s_prime_rewards,
            probs=probs,
            random_state=random_state
        )

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
