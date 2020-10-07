from typing import Dict, Optional

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.dynamic_programming.policy_evaluation import evaluate_v_pi, evaluate_q_pi
from rl.dynamic_programming.policy_improvement import improve_policy_with_v_pi, improve_policy_with_q_pi
from rl.environments.mdp import MdpEnvironment
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=4, page=80)
def iterate_policy_v_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: float,
        update_in_place: bool
) -> Dict[MdpState, float]:
    """
    Run policy iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param theta: See `evaluate_v_pi`.
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-value estimates.
    """

    v_pi: Optional[Dict[MdpState, float]] = None
    improving = True
    i = 0
    while improving:

        print(f'Policy iteration {i + 1}:  ', end='')

        v_pi = evaluate_v_pi(
            agent=agent,
            environment=environment,
            theta=theta,
            num_iterations=None,
            update_in_place=update_in_place,
            initial_v_S=v_pi
        )

        improving = improve_policy_with_v_pi(
            agent=agent,
            environment=environment,
            v_pi=v_pi
        )

        i += 1

    print(f'Policy iteration terminated after {i} iteration(s).\n')

    return v_pi


@rl_text(chapter=4, page=80)
def iterate_policy_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: float,
        update_in_place: bool
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Run policy iteration on an agent using state-value estimates.

    :param agent: Agent.
    :param environment: Environment.
    :param theta: See `evaluate_q_pi`.
    :param update_in_place: See `evaluate_q_pi`.
    :return: Final state-action value estimates.
    """

    q_pi: Optional[Dict[MdpState, Dict[Action, float]]] = None
    improving = True
    i = 0
    while improving:

        print(f'Policy iteration {i + 1}:  ', end='')

        q_pi = evaluate_q_pi(
            agent=agent,
            environment=environment,
            theta=theta,
            num_iterations=None,
            update_in_place=update_in_place,
            initial_q_S_A=q_pi
        )

        improving = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi
        )

        i += 1

    print(f'Policy iteration terminated after {i} iteration(s).\n')

    return q_pi
