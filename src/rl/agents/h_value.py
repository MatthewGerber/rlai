import math
from argparse import Namespace, ArgumentParser
from typing import List, Dict, Tuple

from numpy.random import RandomState

from rl.actions.base import Action
from rl.agents.base import Agent
from rl.utils import IncrementalSampleAverager


class PreferenceGradient(Agent):
    """
    A preference-gradient agent.
    """

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parsed_args, unparsed_args = super().parse_arguments(args)

        parser = ArgumentParser(allow_abbrev=False)

        parser.add_argument(
            '--step-size-alpha',
            type=float,
            default=0.1,
            help='Step-size parameter used to update action preferences.'
        )

        parser.add_argument(
            '--reward-average-alpha',
            type=float,
            default=None,
            help='Constant step-size for reward averaging. If provided, the reward average becomes a recency-weighted average (good for nonstationary environments). If `None` is passed, then the unweighted sample average will be used (good for stationary environments).'
        )

        parser.add_argument(
            '--use-reward-baseline',
            action='store_true',
            help='Whether or not to use a reward baseline when updating action preferences.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

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
                name=f'preference gradient (step size={parsed_args.step_size_alpha})',
                random_state=random_state,
                **dict(parsed_args._get_kwargs())
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

        if self.use_reward_baseline:
            preference_update = self.step_size_alpha * (r - self.R_bar.get_value())
        else:
            preference_update = self.step_size_alpha * r

        self.H_t_A.update({
            a: h_t_a + preference_update * (1 - self.Pr_A[a]) if a == self.most_recent_action else h_t_a - preference_update * self.Pr_A[a]
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
            step_size_alpha: float,
            reward_average_alpha: float,
            use_reward_baseline: bool
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of the agent.
        :param random_state: Random State.
        :param step_size_alpha: Step-size parameter used to update action preferences.
        :param reward_average_alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        :param use_reward_baseline: Whether or not to use a reward baseline when updating action preferences.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state
        )

        self.step_size_alpha = step_size_alpha
        self.use_reward_baseline = use_reward_baseline

        self.H_t_A: Dict[Action, float] = {
            a: 0.0
            for a in self.AA
        }

        self.Pr_A: Dict[Action, float] = dict()

        self.R_bar = IncrementalSampleAverager(
            initial_value=0.0,
            alpha=reward_average_alpha
        )
