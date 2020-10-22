from abc import ABC
from argparse import Namespace, ArgumentParser
from importlib import import_module
from typing import List, Tuple

import numpy as np
from numpy.random import RandomState

from rl.actions import Action
from rl.agents import Agent
from rl.environments.mdp import MdpEnvironment
from rl.states.mdp import MdpState
from rl.utils import sample_list_item


class MdpAgent(Agent, ABC):
    """
    MDP agent. Adds the concepts of state, reward discounting, and policy-based action to the base agent.
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
            '--gamma',
            type=float,
            default=0.1,
            help='Discount factor.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

        return parsed_args, unparsed_args

    @staticmethod
    def parse_mdp_solver_args(
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments for the specified MDP solver.

        :param args: Aguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parser = ArgumentParser(allow_abbrev=False)

        parser.add_argument(
            '--mdp-solver',
            type=str,
            help='Fully-qualified name of the MDP solver function to use.'
        )

        parser.add_argument(
            '--theta',
            type=float,
            help='Minimum tolerated change in value estimates, below which policy evaluation terminates.'
        )

        parser.add_argument(
            '--update-in-place',
            action='store_true',
            help='Update the policy in place (usually quicker).'
        )

        parser.add_argument(
            '--evaluation-iterations-per-improvement',
            type=int,
            help='Number of policy evaluation iterations to execute for each iteration of improvement.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(args)

        return parsed_args, unparsed_args
    
    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            SS: List[MdpState],
            gamma: float
    ):
        """
        Initialize the agent.

        :param AA: List of all possible actions, with identifiers sorted in increasing order from zero.
        :param name: Name of the agent.
        :param random_state: Random state.
        :param SS: List of all possible states, with identifiers sorted in increasing order from zero.
        :param gamma: Discount.
        """

        for i, s in enumerate(SS):
            if s.i != i:
                raise ValueError('States must be sorted in increasing order from zero.')

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state
        )

        self.SS = SS
        self.gamma = gamma

        self.pi = {
            s: {
                a: np.nan
                for a in s.AA
            }
            for s in self.SS
        }


class StochasticMdpAgent(MdpAgent):
    """
    Stochastic MDP agent. Adds random select of action based on probabilities specified in the agent's policy.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment,
            random_state: RandomState
    ) -> Tuple[List[Agent], List[str]]:
        """
        Initialize a list of agents from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :param random_state: Random state.
        :return: 2-tuple of a list of agents and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        agents = [
            StochasticMdpAgent(
                AA=environment.AA,
                name=f'stochastic (gamma={parsed_args.gamma})',
                random_state=random_state,
                SS=environment.SS,
                gamma=parsed_args.gamma
            )
        ]

        # get the mdp solver and arguments
        parsed_mdp_args, unparsed_args = cls.parse_mdp_solver_args(unparsed_args)
        solver_module_name, solver_function_name = parsed_mdp_args.mdp_solver.rsplit('.', maxsplit=1)
        solver_function_module = import_module(solver_module_name)
        solver_function = getattr(solver_function_module, solver_function_name)
        solver_function_arguments = {
            k: v
            for k, v in dict(parsed_mdp_args._get_kwargs()).items()
            if k != 'mdp_solver' and v is not None
        }

        # have each agent solve the mdp with the specified function/parameters
        for agent in agents:
            solver_function(
                agent,
                **solver_function_arguments
            )

        return agents, unparsed_args

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Act randomly.

        :param t: Time tick.
        :return: Action.
        """

        self.most_recent_state: MdpState

        # if the policy is not defined for the most recent state, then update the policy in the most recent state to be
        # uniform across feasible actions. act accordingly
        if self.most_recent_state not in self.pi:
            self.pi[self.most_recent_state] = {
                a: 1 / len(self.most_recent_state.AA)
                for a in self.most_recent_state.AA
            }

        action_prob = self.pi[self.most_recent_state]
        actions = list(action_prob.keys())
        probs = np.array(list(action_prob.values()))

        return sample_list_item(
            x=actions,
            probs=probs,
            random_state=self.random_state
        )

    def reward(
            self,
            r: float
    ):
        """
        Reward the agent.

        :param r: Reward.
        """

        super().reward(
            r=r
        )

    def __init__(
            self,
            AA: List[Action],
            name: str,
            random_state: RandomState,
            SS: List[MdpState],
            gamma: float
    ):
        """
        Initialize the agent with an equiprobable policy over actions.

        :param AA: List of all possible actions, with identifiers sorted in increasing order from zero.
        :param name: Name of the agent.
        :param random_state: Random state.
        :param SS: List of all possible states, with identifiers sorted in increasing order from zero.
        :param gamma: Discount.
        """

        super().__init__(
            AA=AA,
            name=name,
            random_state=random_state,
            SS=SS,
            gamma=gamma
        )

        self.pi = {
            s: {
                a: 1 / len(s.AA)
                for a in s.AA
            }
            for s in self.SS
        }
