from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.states.mdp import MdpState
from rlai.utils import sample_list_item, import_function


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

    def initialize_equiprobable_policy(
            self,
            SS: List[MdpState]
    ):
        """
        Initialize the policy of the current agent to be equiprobable over all actions in a list of states. This is
        useful for environments in which the list of states can be easily enumerated. It is not useful for environments
        (e.g., `rlai.environments.mancala.Mancala`) in which the list of states is very large. The latter problems should
        be addressed with a lazy-expanding list of states (see Mancala for an example).

        :param SS: List of states.
        """

        self.pi = {
            s: {
                a: 1 / len(s.AA)
                for a in s.AA
            }
            for s in SS
        }

    def solve_mdp(
            self
    ):
        """
        Solve the current agent's MDP using the function and arguments passed to the constructor (e.g., from the
        command line).
        """

        self.solver_function(self, **self.solver_function_args)

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            gamma: float,
            solver_function: Optional[Callable] = None,
            solver_function_args: Optional[Dict] = None
    ):
        """
        Initialize the agent with an empty policy. Call `initialize_equiprobable_policy` to initialize the policy for
        a list of states.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param gamma: Discount.
        :param solver_function: Solver function. Required in order to call `self.solve_mdp`.
        :param solver_function_args: Solver function arguments. Required in order to call `self.solve_mdp`.
        """

        super().__init__(
            name=name,
            random_state=random_state
        )

        self.gamma = gamma
        self.solver_function = solver_function
        self.solver_function_args = solver_function_args

        self.pi: Dict[MdpState, Dict[Action, float]] = {}


class StochasticMdpAgent(MdpAgent):
    """
    Stochastic MDP agent. Adds random select of action based on probabilities specified in the agent's policy.
    """

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

        # get the mdp solver and arguments
        parsed_mdp_args, unparsed_args = cls.parse_mdp_solver_args(args)
        solver_function = import_function(parsed_mdp_args.mdp_solver)
        solver_function_arguments = {
            k: v
            for k, v in dict(parsed_mdp_args._get_kwargs()).items()
            if k != 'mdp_solver' and v is not None
        }

        parsed_args, unparsed_args = cls.parse_arguments(unparsed_args)

        agents = [
            StochasticMdpAgent(
                name=f'stochastic (gamma={parsed_args.gamma})',
                random_state=random_state,
                gamma=parsed_args.gamma,
                solver_function=solver_function,
                solver_function_args=solver_function_arguments
            )
        ]

        return agents, unparsed_args

    def __act__(
            self,
            t: int
    ) -> Action:
        """
        Act stochastically according to the policy.

        :param t: Time tick.
        :return: Action.
        """

        self.most_recent_state: MdpState

        # if the policy is not defined for the most recent state, then update the policy in the most recent state to be
        # uniform across feasible actions. act accordingly.
        if self.most_recent_state not in self.pi:
            self.pi[self.most_recent_state] = {
                a: 1 / len(self.most_recent_state.AA)
                for a in self.most_recent_state.AA
            }

        # sample action according to policy for most recent state
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
            name: str,
            random_state: RandomState,
            gamma: float,
            solver_function: Optional[Callable] = None,
            solver_function_args: Optional[Dict] = None
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param gamma: Discount.
        :param solver_function: Solver function. Required in order to call `self.solve_mdp`.
        :param solver_function_args: Solver function arguments. Required in order to call `self.solve_mdp`.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            gamma=gamma,
            solver_function=solver_function,
            solver_function_args=solver_function_args
        )


class Human(MdpAgent):
    """
    An interactive, human-driven agent that prompts for actions at each time step.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> List:
        pass

    def __act__(
            self,
            t: int
    ) -> Action:

        action = None

        while action is None:

            prompt = 'Please select from the following actions:  '

            self.most_recent_state: MdpState

            a_name_i = {
                a.name: i
                for i, a in enumerate(self.most_recent_state.AA)
            }

            for i, name in enumerate(sorted(a_name_i.keys())):
                prompt += f'{", " if i > 0 else ""}{name}'

            prompt += '\nEnter your selection:  '

            try:
                chosen_name = input(prompt)
                action = self.most_recent_state.AA[a_name_i[chosen_name]]
            except Exception:
                pass

        return action

    def reward(
            self,
            r: float):
        pass

    def __init__(
            self
    ):
        super().__init__(
            name='human',
            random_state=None,
            gamma=1
        )
