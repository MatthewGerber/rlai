from typing import Dict

import numpy as np

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.environments.mdp import MdpEnvironment
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=4, page=74)
def iterative_policy_evaluation_of_state_value(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: float,
        update_in_place: bool
) -> Dict[MdpState, float]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Prediction accuracy requirement.
    :param update_in_place: Whether or not to update value estimates in place.
    :return: Dictionary of MDP states and their estimated values.
    """

    if theta <= 0:
        raise ValueError('theta must be > 0.0')

    V_S = np.array([0.0] * len(agent.SS))

    delta = theta + 1
    iterations_finished = 0
    while delta > theta:

        if iterations_finished % 10 == 0:
            print(f'Finished {iterations_finished} iterations:  delta={delta}')

        if update_in_place:
            V_S_to_update = V_S
        else:
            V_S_to_update = np.zeros_like(V_S)

        delta = 0.0

        for s_i, s in enumerate(agent.SS):

            prev_v = V_S[s_i]

            # calculate expected value of current state using current estimates of successor-state values
            new_v = np.sum([

                agent.pi[s][a] * s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * V_S[s_prime_i])

                for a in agent.AA
                for s_prime_i, s_prime in enumerate(agent.SS)
                for r in environment.RR
            ])

            V_S_to_update[s_i] = new_v

            delta = max(delta, abs(prev_v - new_v))

        if not update_in_place:
            V_S = V_S_to_update

        iterations_finished += 1

    print(f'Evaluation completed after {iterations_finished} iteration(s).')

    return {
        s: v
        for s, v in zip(agent.SS, V_S)
    }


@rl_text(chapter=4, page=76)
def iterative_policy_evaluation_of_action_value(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: float,
        update_in_place: bool
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Prediction accuracy requirement.
    :param update_in_place: Whether or not to update value estimates in place.
    :return: Dictionary of MDP states, actions, and their estimated values.
    """

    if theta <= 0:
        raise ValueError('theta must be > 0.0')

    Q_S_A = {
        s: np.array([0.0] * len(agent.AA))
        for s in agent.SS
    }

    delta = theta + 1
    iterations_finished = 0
    while delta > theta:

        if iterations_finished % 10 == 0:
            print(f'Finished {iterations_finished} iterations:  delta={delta}')

        if update_in_place:
            Q_S_A_to_update = Q_S_A
        else:
            Q_S_A_to_update = {
                s: np.zeros_like(Q_S_A[s])
                for s in agent.SS
            }

        delta = 0.0

        for s in agent.SS:

            Q_s_A = Q_S_A[s]

            for a_i, a in enumerate(agent.AA):

                prev_q = Q_s_A[a_i]

                # calculate expected value of current action using current estimates of successor-state action values
                new_q = np.sum([

                    s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * np.sum([
                        agent.pi[s_prime][a_prime] * Q_S_A[s_prime][a_prime_i]
                        for a_prime_i, a_prime in enumerate(agent.AA)
                     ]))

                    for s_prime_i, s_prime in enumerate(agent.SS)
                    for r in environment.RR
                ])

                Q_S_A_to_update[s][a_i] = new_q

                delta = max(delta, abs(prev_q - new_q))

        if not update_in_place:
            Q_S_A = Q_S_A_to_update

        iterations_finished += 1

    print(f'Evaluation completed after {iterations_finished} iteration(s).')

    return {
        s: {
            a: v
            for a, v in zip(agent.AA, Q_S_A[s])
        }
        for s in agent.SS
    }
