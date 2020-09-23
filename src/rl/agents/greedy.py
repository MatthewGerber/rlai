from argparse import Namespace, ArgumentParser
from typing import List, Tuple

from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.base import Agent
from rl.agents.nonassociative import Nonassociative
from rl.meta import rl_text


@rl_text(chapter=2, page=27)
class EpsilonGreedy(Nonassociative):
    """
    An epsilon-greedy agent.
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
            '--epsilon',
            type=float,
            nargs='+',
            help='Space-separated list of epsilon values to evaluate.'
        )

        parser.add_argument(
            '--epsilon-reduction-rate',
            type=float,
            default=0.0,
            help='Percentage reduction of epsilon from its initial value. This is applied at each time step when the agent explores. For example, pass 0 for no reduction and 0.01 for a 1-percent reduction at each exploration step.'
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

        # grab and delete epsilons from parsed arguments
        epsilons = parsed_args.epsilon
        del parsed_args.epsilon

        # initialize agents
        agents = [
            EpsilonGreedy(
                AA=AA,
                name=f'epsilon-greedy (e={epsilon:0.2f})',
                random_state=random_state,
                epsilon=epsilon,
                **dict(parsed_args._get_kwargs())
            )
            for epsilon in epsilons
        ]

        return agents, unparsed_args

    def reset_for_new_run(
            self
    ):
        """
        Reset the agent to a state prior to any learning.
        """

        super().reset_for_new_run()

        self.epsilon = self.original_epsilon
        self.greedy_action = list(self.Q.keys())[0]

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Act in an epsilon-greedy fashion.

        :param t: Current time step.
        :return: Action.
        """

        if self.random_state.random_sample() < self.epsilon:
            a = self.random_state.choice(self.AA)
            self.epsilon *= (1 - self.epsilon_reduction_rate)
        else:
            a = self.greedy_action

        return a

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward value.
        """

        super().reward(r)

        # get new greedy action, which might have changed
        self.greedy_action = max(self.Q.items(), key=lambda action_value: action_value[1].get_value())[0]

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            initial_q_value: float,
            alpha: float,
            epsilon: float,
            epsilon_reduction_rate: float
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions.
        :param name: Name of agent.
        :param random_state: Random state.
        :param initial_q_value: Initial Q-value to use for all actions. Use values greater than zero to encourage
        exploration in the early stages of the run.
        :param alpha: Step-size parameter for incremental reward averaging. See `IncrementalSampleAverager` for details.
        :param epsilon: Epsilon.
        :param epsilon_reduction_rate: Rate at which `epsilon` is reduced from its initial value to zero. This is the
        percentage reduction, and it is applied at each time step when the agent explores. For example, pass 0 for no
        reduction and 0.01 for a 1-percent reduction at each exploration step.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state,
            initial_q_value=initial_q_value,
            alpha=alpha
        )

        self.epsilon = self.original_epsilon = epsilon
        self.epsilon_reduction_rate = epsilon_reduction_rate
        self.greedy_action = None
