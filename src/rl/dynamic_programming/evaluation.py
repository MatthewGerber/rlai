from typing import Dict

import numpy as np

from rl.agents.mdp import MdpAgent
from rl.environments.mdp import MdpEnvironment
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=4, page=74)
def iterative_policy_evaluation(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: float,
        update_in_place: bool
) -> Dict[MdpState, float]:
    """
    Perform iterative policy evaluation on an agent's policy within an environment.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Prediction accuracy requirement.
    :param update_in_place: Whether or not to update value estimates in place.
    :return: Dictionary of MDP states and their estimated values.
    """

    if theta <= 0:
        raise ValueError('Theta must be > 0.')

    V_S = np.array([0.0] * len(environment.SS))

    delta = theta + 1
    s: MdpState
    iterations_finished = 0
    while delta > theta:

        if iterations_finished % 10 == 0:
            print(f'Finished {iterations_finished} iterations:  delta={delta}')

        if update_in_place:
            V_S_to_update = V_S
        else:
            V_S_to_update = np.zeros_like(V_S)

        delta = 0.0

        for s_i, s in enumerate(environment.SS):

            prev_v = V_S[s_i]

            # calculate expected value of current state using current estimates of successor-state values
            new_v = np.sum([

                agent.pi[s][a] * s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * V_S[s_prime_i])

                for a in environment.AA
                for s_prime_i, s_prime in enumerate(environment.SS)
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
        for s, v in zip(environment.SS, V_S)
    }
