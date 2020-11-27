from abc import ABC
from argparse import Namespace, ArgumentParser
from typing import List, Tuple, Optional, Dict, Union

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.states.mdp import MdpState
from rlai.utils import sample_list_item


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
            '--continuous-state-discretization-resolution',
            type=float,
            help='Continuous-state discretization resolution.'
        )

        parser.add_argument(
            '--gamma',
            type=float,
            help='Discount factor.'
        )

        parsed_args, unparsed_args = parser.parse_known_args(unparsed_args, parsed_args)

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

    def get_state_i(
            self,
            state_descriptor: Union[str, np.ndarray]
    ) -> int:
        """
        Get the integer identifier for a state. The returned value is guaranteed to be the same for the same state,
        both throughout the life of the current agent as well as after the current agent has been pickled for later
        use (e.g., in checkpoint-based resumption).

        :param state_descriptor: State descriptor, either a string (for discrete states) or an array representing a
        position within an n-dimensional continuous state space.

        :return: Integer identifier.
        """

        if isinstance(state_descriptor, np.ndarray):

            if self.continuous_state_discretization_resolution is None:
                raise ValueError('Attempted to discretize a continuous state without a resolution.')

            state_descriptor = '|'.join(
                str(int(state_dim_value / self.continuous_state_discretization_resolution))
                for state_dim_value in state_descriptor
            )

        elif not isinstance(state_descriptor, str):
            raise ValueError(f'Unknown state space type:  {type(state_descriptor)}')

        if state_descriptor not in self.state_id_str_int:
            self.state_id_str_int[state_descriptor] = len(self.state_id_str_int)

        return self.state_id_str_int[state_descriptor]

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            continuous_state_discretization_resolution: Optional[float],
            gamma: float
    ):
        """
        Initialize the agent with an empty policy. Call `initialize_equiprobable_policy` to initialize the policy for
        a list of states.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param continuous_state_discretization_resolution: A discretization resolution for continuous-state
        environments. Providing this value allows the agent to be used with discrete-state methods via
        discretization of the continuous-state dimensions.
        :param gamma: Discount.
        """

        super().__init__(
            name=name,
            random_state=random_state
        )

        self.continuous_state_discretization_resolution = continuous_state_discretization_resolution
        self.gamma = gamma

        self.pi: Dict[MdpState, Dict[Action, float]] = {}
        self.state_id_str_int: Dict[str, int] = {}


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

        parsed_args, unparsed_args = cls.parse_arguments(args)

        agents = [
            StochasticMdpAgent(
                name=f'stochastic (gamma={parsed_args.gamma})',
                random_state=random_state,
                **vars(parsed_args)
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
            continuous_state_discretization_resolution: Optional[float],
            gamma: float
    ):
        """
        Initialize the agent.

        :param name: Name of the agent.
        :param random_state: Random state.
        :param continuous_state_discretization_resolution: A discretization resolution for continuous-state
        environments. Providing this value allows the agent to be used with discrete-state methods via
        discretization of the continuous-state dimensions.
        :param gamma: Discount.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            continuous_state_discretization_resolution=continuous_state_discretization_resolution,
            gamma=gamma
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
            continuous_state_discretization_resolution=None,
            gamma=1
        )
