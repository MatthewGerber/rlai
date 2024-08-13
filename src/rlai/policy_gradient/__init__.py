from argparse import ArgumentParser
from typing import List, Tuple, Optional

from numpy.random import RandomState

from rlai.core import Agent, StochasticMdpAgent, Environment, MdpState, State
from rlai.docs import rl_text
from rlai.policy_gradient.policies import ParameterizedPolicy
from rlai.state_value import StateValueEstimator
from rlai.utils import parse_arguments, load_class


@rl_text(chapter='Agents', page=1)
class ParameterizedMdpAgent(StochasticMdpAgent):
    """
    A stochastic MDP agent whose policy is directly parameterized. This agent is generally appropriate when both the
    state and action spaces are continuous. If the action space is discrete, then consider the `ActionValueMdpAgent`.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--policy',
            type=str,
            help='Fully-qualified type name of policy to use.'
        )

        parser.add_argument(
            '--v-S',
            type=str,
            help='Fully-qualified type name of baseline state-value estimator to use, or ignore for no baseline.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: Environment
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize an MDP agent from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # load state-value estimator, which is optional.
        v_S = None
        if parsed_args.v_S is not None:
            estimator_class = load_class(parsed_args.v_S)
            v_S, unparsed_args = estimator_class.init_from_arguments(
                args=unparsed_args,
                random_state=random_state,
                environment=environment
            )
        del parsed_args.v_S

        # load parameterized policy
        policy_class = load_class(parsed_args.policy)
        policy, unparsed_args = policy_class.init_from_arguments(
            args=unparsed_args,
            environment=environment
        )
        del parsed_args.policy

        # noinspection PyUnboundLocalVariable
        agent = cls(
            name=f'parameterized (gamma={parsed_args.gamma})',
            random_state=random_state,
            pi=policy,
            v_S=v_S,
            **vars(parsed_args)
        )

        return [agent], unparsed_args

    def reset_for_new_run(
            self,
            state: State
    ):
        """
        Reset for new run.
        """

        super().reset_for_new_run(state)

        if self.v_S is not None:
            assert isinstance(state, MdpState)
            self.v_S.reset_for_new_run(state)

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            pi: ParameterizedPolicy,
            gamma: float,
            v_S: Optional[StateValueEstimator]
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param pi: Policy.
        :param gamma: Discount.
        :param v_S: Baseline state-value estimator.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            pi=pi,
            gamma=gamma
        )

        self.v_S = v_S
