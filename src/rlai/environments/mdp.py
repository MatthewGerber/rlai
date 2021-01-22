from abc import ABC, abstractmethod
from argparse import ArgumentParser
from copy import copy
from functools import partial
from queue import PriorityQueue
from typing import List, Tuple, Optional, final, Dict

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Agent
from rlai.agents.mdp import MdpAgent
from rlai.environments import Environment
from rlai.meta import rl_text
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.rewards import Reward
from rlai.runners.monitor import Monitor
from rlai.states import State
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager, sample_list_item, parse_arguments, get_base_argument_parser


@rl_text(chapter=3, page=47)
class MdpEnvironment(Environment, ABC):
    """
    MDP environment.
    """

    def reset_for_new_run(
            self,
            agent: MdpAgent
    ) -> State:
        """
        Reset the the environment to a random nonterminal state, if any are specified, or to None.

        :param agent: Agent used to generate on-the-fly state identifiers.
        """

        super().reset_for_new_run(agent)

        if len(self.nonterminal_states) > 0:
            self.state = self.random_state.choice(self.nonterminal_states)
        else:
            self.state = None

        return self.state

    @abstractmethod
    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance from the current state given an action.

        :param state: State to advance.
        :param t: Current time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of next state and next reward.
        """

    @final
    def run_step(
            self,
            t: int,
            agent: Agent,
            monitor: Monitor
    ) -> bool:
        """
        Run a step of the environment with an agent.

        :param t: Step.
        :param agent: Agent.
        :param monitor: Monitor.
        :return: True if a terminal state was entered and the run should terminate, and False otherwise.
        """

        a = agent.act(t=t)

        self.state, next_reward = self.advance(
            state=self.state,
            t=t,
            a=a,
            agent=agent
        )

        agent.sense(
            state=self.state,
            t=t+1
        )

        agent.reward(next_reward.r)

        monitor.report(t=t+1, action_reward=next_reward.r)

        return self.state.terminal

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            SS: Optional[List[MdpState]] = None,
            RR: Optional[List[Reward]] = None
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param SS: Prespecified list of states, or None for no prespecification.
        :param RR: Prespecified list of rewards, or None for no prespecification.
        """

        if SS is None:
            SS = []

        if RR is None:
            RR = []

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.SS = SS
        self.RR = RR

        self.terminal_states = [s for s in self.SS if s.terminal]
        self.nonterminal_states = [s for s in self.SS if not s.terminal]
        self.state: Optional[MdpState] = None


@rl_text(chapter=3, page=48)
class ModelBasedMdpEnvironment(MdpEnvironment, ABC):
    """
    Model-based MDP environment. Adds the specification of a probability distribution over next states and rewards.
    """

    def check_marginal_probabilities(
            self
    ):
        """
        Check the marginal next-state and next-reward probabilities, to ensure they sum to 1. Raises an exception if
        this is not the case.
        """

        for s in self.p_S_prime_R_given_S_A:
            for a in self.p_S_prime_R_given_S_A[s]:

                marginal_prob = sum([
                    self.p_S_prime_R_given_S_A[s][a][s_prime][r]
                    for s_prime in self.p_S_prime_R_given_S_A[s][a]
                    for r in self.p_S_prime_R_given_S_A[s][a][s_prime]
                ])

                if marginal_prob != 1.0:
                    raise ValueError(f'Expected next-state/next-reward marginal probability of 1.0, but got {marginal_prob}.')

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: Agent
    ) -> Tuple[MdpState, Reward]:
        """
        Advance from the current state given an action, based on the current state's model probability distribution.

        :param state: State to advance.
        :param t: Current time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of next state and next reward.
        """

        # get next-state / reward tuples
        s_prime_rewards = [
            (s_prime, reward)
            for s_prime in self.p_S_prime_R_given_S_A[state][a]
            for reward in self.p_S_prime_R_given_S_A[state][a][s_prime]
            if self.p_S_prime_R_given_S_A[state][a][s_prime][reward] > 0.0
        ]

        # get probability of each tuple
        probs = np.array([
            self.p_S_prime_R_given_S_A[state][a][s_prime][reward]
            for s_prime, reward in s_prime_rewards
        ])

        # sample next state and reward
        next_state, next_reward = sample_list_item(
            x=s_prime_rewards,
            probs=probs,
            random_state=self.random_state
        )

        return next_state, next_reward

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            SS: Optional[List[MdpState]] = None,
            RR: Optional[List[Reward]] = None
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param SS: Prespecified list of states, or None for no prespecification.
        :param RR: Prespecified list of rewards, or None for no prespecification.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T,
            SS=SS,
            RR=RR
        )

        # initialize an empty model
        self.p_S_prime_R_given_S_A: Dict[
            MdpState, Dict[
                Action, Dict[
                    MdpState, Dict[
                        Reward, float
                    ]
                ]
            ]
        ] = {
            s: {
                a: {
                    s: {
                        r: 0.0
                        for r in self.RR
                    }
                    for s in self.SS
                }
                for a in s.AA
            }
            for s in self.SS
        }


@rl_text(chapter=8, page=159)
class MdpPlanningEnvironment(MdpEnvironment, ABC):
    """
    An MDP planning environment, used to generate simulated experience based on a model of the MDP that is learned
    through direct experience with the actual environment.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        # don't use super's argument parser, so that we do not pick up the --T argument intended for the actual
        # environment. we're going to use --T-planning instead (see below).

        parser = get_base_argument_parser()

        parser.add_argument(
            '--T-planning',
            type=int,
            help='Maximum number of planning time steps to run.'
        )

        parser.add_argument(
            '--num-planning-improvements-per-direct-improvement',
            type=int,
            help='Number of planning improvements to make for each direct improvement.'
        )

        return parser

    def reset_for_new_run(
            self,
            agent: Agent
    ) -> Optional[State]:
        """
        Reset the planning environment to a randomly sampled state.

        :param agent: Agent.
        :return: New state.
        """

        self.state = self.model.sample_state(self.random_state)

        return self.state

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            model: StochasticEnvironmentModel,
            num_planning_improvements_per_direct_improvement: int
    ):
        """
        Initialize the planning environment.

        :param name: Name of the environment.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param model: Model to be learned from direct experience for the purpose of planning from simulated experience.
        :param num_planning_improvements_per_direct_improvement: Number of planning improvements to make for each
        improvement based on actual experience. Pass None for no planning.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.model = model
        self.num_planning_improvements_per_direct_improvement = num_planning_improvements_per_direct_improvement


@rl_text(chapter=8, page=168)
class PrioritizedSweepingMdpPlanningEnvironment(MdpPlanningEnvironment):
    """
    State-action transitions are prioritized based on the degree to which learning updates their values, and transitions
    with the highest priority are explored during planning.
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
            '--priority-theta',
            type=float,
            help='Threshold on priority values, below which state-action pairs are added to the priority queue. Note that all priority values are negative, with higher priorities being larger negative values.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        planning_environment = PrioritizedSweepingMdpPlanningEnvironment(
            name=f"prioritized planning (theta={parsed_args.priority_theta})",
            random_state=random_state,
            model=StochasticEnvironmentModel(),
            **vars(parsed_args)
        )

        return planning_environment, unparsed_args

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: MdpAgent
    ) -> Tuple[Tuple[MdpState, Action, MdpState], Reward]:
        """
        Advance a planning state based on priorities.

        :param state: State to be advanced.
        :param t: Time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of (1) 3-tuple of current state, current action, and next-state, and (2) reward.
        """

        planning_state, planning_a = self.get_state_action_with_highest_priority()

        # if there is nothing left in the priority queue, then the planning episode is done.
        if planning_state is None:
            planning_state = state
            planning_a = a
            next_state = copy(state)
            next_state.terminal = True
            r = 0.0
        else:

            # sample next state and reward from model
            next_state, r = self.model.sample_next_state_and_reward(planning_state, planning_a, self.random_state)

            # add predecessors into priority queue
            for pred_state, pred_action, r in self.get_predecessor_state_action_rewards(planning_state):
                target_value = r + agent.gamma * self.bootstrap_function(state=planning_state, t=t+1)[0]
                priority = -abs(target_value - self.q_S_A[pred_state][pred_action].get_value())
                self.add_state_action_priority(pred_state, pred_action, priority)

        return (planning_state, planning_a, next_state), Reward(None, r)

    def get_predecessor_state_action_rewards(
            self,
            state: MdpState
    ) -> List[Tuple[MdpState, Action, float]]:
        """
        Get a list of predecessor state-action-reward tuples for a given state.

        :param state: State.
        :return: List of predecessor state-action-reward 3-tuples for a given state.
        """

        return [
            (
                pred_state,
                pred_action,
                self.model.state_reward_averager[state].get_value()
            )
            for pred_state in self.model.state_action_next_state_count
            for pred_action in self.model.state_action_next_state_count[pred_state]
            if state in self.model.state_action_next_state_count[pred_state][pred_action]
        ]

    def add_state_action_priority(
            self,
            state: MdpState,
            action: Action,
            priority: float
    ):
        """
        Add a state-action priority.

        :param state: State.
        :param action: Action.
        :param priority: Priority. Lower numbers are higher priority.
        """

        if self.priority_theta is None or priority < self.priority_theta:

            # use counter to break all ties
            self.num_priorities += 1

            self.state_action_priority.put((priority, self.num_priorities, (state, action)))

    def get_state_action_with_highest_priority(
            self
    ) -> Tuple[Optional[MdpState], Optional[Action]]:
        """
        Get the state-action pair with the highest priority.

        :return: 2-tuple of state-action pair, or (None, None) if the priority queue is empty.
        """

        if self.state_action_priority.empty():
            return None, None
        else:
            return self.state_action_priority.get()[2]

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            model: StochasticEnvironmentModel,
            num_planning_improvements_per_direct_improvement: int,
            priority_theta: Optional[float],
            T_planning: int
    ):
        """
        Initialize the planning environment.

        :param name: Name of the environment.
        :param random_state: Random state.
        :param model: Model to be learned from direct experience for the purpose of planning from simulated experience.
        :param num_planning_improvements_per_direct_improvement: Number of planning improvements to run per direct
        improvement.
        :param priority_theta: Priority threshold, below which state-action pairs are added to the priority queue for
        exploration during planning-based learning. Pass None for no threshold (accept all state-action pairs).
        :param T_planning: Maximum number of planning time steps to run. Prioritized sweeping can easily get into
        situations of infinite episode length, since the planning episodes are not generated from epsilon-greedy
        polices.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T_planning,
            model=model,
            num_planning_improvements_per_direct_improvement=num_planning_improvements_per_direct_improvement
        )

        self.priority_theta = priority_theta

        self.state_action_priority: PriorityQueue = PriorityQueue()
        self.num_priorities = 0
        self.bootstrap_function: Optional[partial] = None
        self.q_S_A: Optional[Dict[MdpState, Dict[Action, IncrementalSampleAverager]]] = None


@rl_text(chapter=8, page=174)
class TrajectorySamplingMdpPlanningEnvironment(MdpPlanningEnvironment):
    """
    State-action transitions are selected by the agent based on the agent's policy, and the selected transitions are
    explored during planning.
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

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        planning_environment = TrajectorySamplingMdpPlanningEnvironment(
            name=f"trajectory planning",
            random_state=random_state,
            model=StochasticEnvironmentModel(),
            **vars(parsed_args)
        )

        return planning_environment, unparsed_args

    def advance(
            self,
            state: MdpState,
            t: int,
            a: Action,
            agent: MdpAgent
    ) -> Tuple[Tuple[MdpState, Action, MdpState], Reward]:
        """
        Advance a planning state.

        :param state: State to be advanced.
        :param t: Time step.
        :param a: Action.
        :param agent: Agent.
        :return: 2-tuple of (1) 3-tuple of current state, current action, and next-state, and (2) reward.
        """

        # sample a random action if the given one is not defined by the model
        if not self.model.is_defined_for_state_action(state, a):
            a = self.model.sample_action(state, self.random_state)

        # sample next state and reward from model
        next_state, r = self.model.sample_next_state_and_reward(state, a, self.random_state)

        return (state, a, next_state), Reward(None, r)

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            model: StochasticEnvironmentModel,
            num_planning_improvements_per_direct_improvement: int,
            T_planning: Optional[int]
    ):
        """
        Initialize the planning environment.

        :param name: Name of the environment.
        :param random_state: Random state.
        :param model: Model to be learned from direct experience for the purpose of planning from simulated experience.
        :param num_planning_improvements_per_direct_improvement: Number of planning improvements to run per direct
        improvement.
        :param T_planning: Maximum number of steps to run, or None for no limit.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T_planning,
            model=model,
            num_planning_improvements_per_direct_improvement=num_planning_improvements_per_direct_improvement
        )
