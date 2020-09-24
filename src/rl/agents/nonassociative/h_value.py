import math
from typing import List, Dict, Tuple

from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.agents.nonassociative.base import Nonassociative
from rl.utils import IncrementalSampleAverager


class PreferenceGradient(Nonassociative):
    """
    A preference-gradient agent.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            AA: List[Action],
            random_state: RandomState
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param AA: List of possible actions.
        :param random_state: Random state.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        # initialize agents
        agents = [
            PreferenceGradient(
                AA=AA,
                name=f'preference gradient (alpha={parsed_args.alpha})',
                random_state=random_state,
                alpha=parsed_args.alpha
            )
        ]

        return agents, unparsed_args

    def reset_for_new_run(
            self
    ):
        """
        Reset the agent to a state prior to any learning.
        """

        super().reset_for_new_run()

        self.H_t_A.update({
            a: 0.0
            for a in self.H_t_A
        })

        self.update_action_probabilities()
        self.R_bar.reset()

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Sample a random action based on current preferences.

        :param t: Time step.
        :return: Action.
        """

        return self.random_state.choice(
            a=self.AA,
            p=[
                self.Pr_A[a]
                for a in self.AA
            ]
        )

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward value.
        """

        super().reward(r)

        self.R_bar.update(r)

        relative_reward_step = self.alpha * (r - self.R_bar.get_value())

        self.H_t_A.update({
            a: h_t_a + relative_reward_step * (1 - self.Pr_A[a]) if a == self.most_recent_action else h_t_a - relative_reward_step * self.Pr_A[a]
            for a, h_t_a in self.H_t_A.items()
        })

        self.update_action_probabilities()

    def update_action_probabilities(
            self
    ):
        """
        Update action probabilities based on current preferences.
        """

        denominator = sum([
            math.exp(h_t_a)
            for h_t_a in self.H_t_A.values()
        ])

        self.Pr_A.update({
            a: math.exp(h) / denominator
            for a, h in self.H_t_A.items()
        })

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            alpha: float
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of the agent.
        :param random_state: Random State.
        :param alpha: Step size.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state,
            alpha=alpha
        )

        self.H_t_A: Dict[Action, float] = {
            a: 0.0
            for a in self.AA
        }

        self.Pr_A: Dict[Action, float] = dict()

        self.R_bar = IncrementalSampleAverager(
            initial_value=0.0,
            alpha=self.alpha
        )
