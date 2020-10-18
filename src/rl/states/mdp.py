from typing import List, Dict

from rl.actions import Action
from rl.rewards import Reward
from rl.states import State


class MdpState(State):
    """
    MDP state.
    """

    def __init__(
            self,
            i: int,
            AA: List[Action],
            RR: List[Reward],
            terminal: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions that can be taken from this state.
        :param RR: All rewards provided by the environment.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=i
        )

        self.AA = AA
        self.RR = RR
        self.terminal = terminal


class ModelBasedMdpState(MdpState):
    """
    Model-based MDP state. Adds the specification of a probability distribution over next states and rewards.
    """

    def check_marginal_probabilities(
            self
    ):
        """
        Check the marginal next-state and -reward probabilities, to ensure they sum to 1. Raises an exception if this is
        not the case.
        """

        # check that marginal probabilities for each state sum to 1
        for a in self.p_S_prime_R_given_A:

            marginal_prob = sum([
                self.p_S_prime_R_given_A[a][s_prime][r]
                for s_prime in self.p_S_prime_R_given_A[a]
                for r in self.p_S_prime_R_given_A[a][s_prime]
            ])

            if marginal_prob != 1.0:
                raise ValueError(f'Expected state-marginal probability of 1.0, got {marginal_prob}.')

    def __init__(
            self,
            i: int,
            AA: List[Action],
            RR: List[Reward],
            terminal: bool
    ):
        """
        Initialize the MDP state.

        :param i: State index.
        :param AA: All actions that can be taken from this state.
        :param RR: All rewards provided by the environment.
        :param terminal: Whether or not the state is terminal.
        """

        super().__init__(
            i=i,
            AA=AA,
            RR=RR,
            terminal=terminal
        )

        # empty list of next states. to be initialized in `init_model`.
        self.SS = []

        # initialize an empty model within the state (see `init_model` for initialization)
        self.p_S_prime_R_given_A: Dict[
            Action, Dict[
                ModelBasedMdpState, Dict[
                    Reward, float
                ]
            ]
        ] = {}

    def init_model(
            self,
            SS: List[State]
    ):
        """
        Initialize the model within each state with zeros.

        :param SS: All states that can follow the current one based on the current state's actions.
        """

        self.SS = SS

        self.p_S_prime_R_given_A = {
            a: {
                s: {
                    r: 0.0
                    for r in self.RR
                }
                for s in self.SS
            }
            for a in self.AA
        }
