import logging
from typing import Dict, Optional

from rlai.core import Action, MdpState, MdpAgent
from rlai.core.environments.mdp import ModelBasedMdpEnvironment
from rlai.docs import rl_text
from rlai.gpi.dynamic_programming.evaluation import evaluate_v_pi, evaluate_q_pi
from rlai.gpi.state_action_value.tabular import TabularPolicy


@rl_text(chapter=4, page=80)
def iterate_policy_v_pi(
        agent: MdpAgent,
        environment: ModelBasedMdpEnvironment,
        theta: float,
        update_in_place: bool
) -> Dict[MdpState, float]:
    """
    Run policy iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates.
    :param update_in_place: Whether to update value estimates in place.
    :return: Final state-value estimates.
    """

    assert isinstance(agent.pi, TabularPolicy)

    v_pi: Optional[Dict[MdpState, float]] = None
    improving = True
    i = 0
    while improving:

        logging.info(f'Policy iteration {i + 1}')

        v_pi, _ = evaluate_v_pi(
            agent=agent,
            environment=environment,
            theta=theta,
            num_iterations=None,
            update_in_place=update_in_place,
            initial_v_S=v_pi
        )

        assert v_pi is not None

        improving = agent.pi.improve_with_v_pi(
            gamma=agent.gamma,
            environment=environment,
            v_pi=v_pi
        ) > 0

        i += 1

    logging.info(f'Policy iteration terminated after {i} iteration(s).')

    assert v_pi is not None

    return v_pi


@rl_text(chapter=4, page=80)
def iterate_policy_q_pi(
        agent: MdpAgent,
        environment: ModelBasedMdpEnvironment,
        theta: float,
        update_in_place: bool
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Run policy iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: Minimum tolerated change in state-value estimates, below which evaluation terminates.
    :param update_in_place: Whether to update value estimates in place.
    :return: Final state-action value estimates.
    """

    assert isinstance(agent.pi, TabularPolicy)

    q_pi: Optional[Dict[MdpState, Dict[Action, float]]] = None
    improving = True
    i = 0
    while improving:

        logging.info(f'Policy iteration {i + 1}')

        q_pi, _ = evaluate_q_pi(
            agent=agent,
            environment=environment,
            theta=theta,
            num_iterations=None,
            update_in_place=update_in_place,
            initial_q_S_A=q_pi
        )

        assert q_pi is not None

        improving = agent.pi.improve_with_q_pi(q_pi) > 0

        i += 1

    logging.info(f'Policy iteration terminated after {i} iteration(s).')

    assert q_pi is not None

    return q_pi


@rl_text(chapter=4, page=82)
def iterate_value_v_pi(
        agent: MdpAgent,
        environment: ModelBasedMdpEnvironment,
        theta: float,
        evaluation_iterations_per_improvement: int,
        update_in_place: bool
) -> Dict[MdpState, float]:
    """
    Run dynamic programming value iteration on an agent using state-value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_v_pi`.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement (e.g., passing 1 results in Equation 4.10).
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
    """

    assert isinstance(agent.pi, TabularPolicy)

    v_pi: Optional[Dict[MdpState, float]] = None
    i = 0
    while True:

        logging.info(f'Value iteration {i + 1}')

        v_pi, delta = evaluate_v_pi(
            agent=agent,
            environment=environment,
            theta=None,
            num_iterations=evaluation_iterations_per_improvement,
            update_in_place=update_in_place,
            initial_v_S=v_pi
        )

        assert v_pi is not None

        agent.pi.improve_with_v_pi(
            gamma=agent.gamma,
            environment=environment,
            v_pi=v_pi
        )

        i += 1

        if delta < theta:
            break

    assert v_pi is not None

    logging.info(f'Value iteration of v_pi terminated after {i} iteration(s).')

    return v_pi


@rl_text(chapter=4, page=84)
def iterate_value_q_pi(
        agent: MdpAgent,
        environment: ModelBasedMdpEnvironment,
        theta: float,
        evaluation_iterations_per_improvement: int,
        update_in_place: bool
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Run value iteration on an agent using state-action value estimates.

    :param agent: MDP agent. Must contain a policy `pi` that has been fully initialized with instances of
    `rlai.core.ModelBasedMdpState`.
    :param environment: Model-based MDP environment to evaluate.
    :param theta: See `evaluate_q_pi`.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement.
    :param update_in_place: See `evaluate_q_pi`.
    :return: Final state-action value estimates.
    """

    assert isinstance(agent.pi, TabularPolicy)

    q_pi: Optional[Dict[MdpState, Dict[Action, float]]] = None
    i = 0
    while True:

        logging.info(f'Value iteration {i + 1}')

        q_pi, delta = evaluate_q_pi(
            agent=agent,
            environment=environment,
            theta=None,
            num_iterations=evaluation_iterations_per_improvement,
            update_in_place=update_in_place,
            initial_q_S_A=q_pi
        )
        assert q_pi is not None
        agent.pi.improve_with_q_pi(q_pi)

        i += 1

        if delta < theta:
            break

    assert q_pi is not None

    logging.info(f'Value iteration of q_pi terminated after {i} iteration(s).')

    return q_pi
