from typing import Dict, Optional

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.dynamic_programming.policy_evaluation import evaluate_v_pi, evaluate_q_pi
from rl.dynamic_programming.policy_improvement import improve_policy_with_v_pi, improve_policy_with_q_pi
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=4, page=82)
def iterate_value_v_pi(
        agent: MdpAgent,
        evaluation_iterations_per_improvement: int,
        update_in_place: bool
) -> Dict[MdpState, float]:
    """
    Run value iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement (e.g., passing 1 results in Equation 4.10).
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
    """

    v_pi: Optional[Dict[MdpState, float]] = None
    improving = True
    i = 0
    while improving:

        print(f'Value iteration {i + 1}:  ', end='')

        v_pi = evaluate_v_pi(
            agent=agent,
            theta=None,
            num_iterations=evaluation_iterations_per_improvement,
            update_in_place=update_in_place,
            initial_v_S=v_pi
        )

        improving = improve_policy_with_v_pi(
            agent=agent,
            v_pi=v_pi
        )

        i += 1

    print(f'Value iteration terminated after {i} iteration(s).')

    return v_pi


@rl_text(chapter=4, page=82)
def iterate_value_q_pi(
        agent: MdpAgent,
        evaluation_iterations_per_improvement: int,
        update_in_place: bool
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Run policy iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param evaluation_iterations_per_improvement: Number of policy evaluation iterations to execute for each iteration
    of improvement (e.g., passing 1 results in Equation 4.10).
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-action value estimates.
    """

    q_pi: Optional[Dict[MdpState, Dict[Action, float]]] = None
    improving = True
    i = 0
    while improving:

        print(f'Value iteration {i + 1}:  ', end='')

        q_pi = evaluate_q_pi(
            agent=agent,
            theta=None,
            num_iterations=evaluation_iterations_per_improvement,
            update_in_place=update_in_place,
            initial_q_S_A=q_pi
        )

        improving = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi
        )

        i += 1

    print(f'Value iteration terminated after {i} iteration(s).')

    return q_pi
