from argparse import Namespace, ArgumentParser
from typing import List, Tuple

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.meta import rl_text
from rlai.states import State
from rlai.utils import IncrementalSampleAverager, sample_list_item


@rl_text(chapter=2, page=37)
class PreferenceGradient(Agent):
    """
    Preference-gradient agent.
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
            random_state: RandomState
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        # initialize agents
        agents = [
            PreferenceGradient(
                name=f'preference gradient (step size={parsed_args.step_size_alpha})',
                random_state=random_state,
                **dict(parsed_args._get_kwargs())
            )
        ]

        return agents, unparsed_args

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset the agent to a state prior to any learning.

        :param state: New state.
        """

        super().reset_for_new_run(state)

        self.H_t_A = np.zeros(len(self.most_recent_state.AA))
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

        return sample_list_item(self.most_recent_state.AA, self.Pr_A, self.random_state)

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward value.
        """

        super().reward(r)

        if self.use_reward_baseline:
            self.R_bar.update(r)
            preference_update = self.step_size_alpha * (r - self.R_bar.get_value())
        else:
            preference_update = self.step_size_alpha * r

        # get preference update for action taken
        most_recent_action_i = self.most_recent_action.i
        update_action_taken = self.H_t_A[most_recent_action_i] + preference_update * (1 - self.Pr_A[most_recent_action_i])

        # get other-action preference update for all actions
        update_all = self.H_t_A - preference_update * self.Pr_A

        # set preferences
        self.H_t_A = update_all
        self.H_t_A[most_recent_action_i] = update_action_taken

        self.update_action_probabilities()

    def update_action_probabilities(
            self
    ):
        """
        Update action probabilities based on current preferences.
        """

        exp_h_t_a = np.e ** self.H_t_A
        exp_h_t_a_sum = exp_h_t_a.sum()

        self.Pr_A = exp_h_t_a / exp_h_t_a_sum

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            step_size_alpha: float,
            use_reward_baseline: bool,
            reward_average_alpha: float
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random State.
        :param step_size_alpha: Step-size parameter used to update action preferences.
        :param use_reward_baseline: Whether or not to use a reward baseline when updating action preferences.
        :param reward_average_alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        """

        super().__init__(
            name=name,
            random_state=random_state
        )

        self.step_size_alpha = step_size_alpha
        self.use_reward_baseline = use_reward_baseline
        self.R_bar = IncrementalSampleAverager(
            initial_value=0.0,
            alpha=reward_average_alpha
        )

        self.H_t_A: np.ndarray = None
        self.Pr_A: np.ndarray = None
