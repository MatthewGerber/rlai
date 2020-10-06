import math
from typing import Optional, Dict

import numpy as np

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.environments.mdp import MdpEnvironment
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=4, page=74)
def evaluate_v_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: Optional[float],
        num_iterations: Optional[int],
        update_in_place: bool,
        initial_V_S: Optional[Dict[MdpState, float]] = None
) -> Dict[MdpState, float]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Minimum tolerated change in state value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_V_S: Initial guess at state-value, or None for no guess.
    :return: Dictionary of MDP states and their estimated values.
    """

    # treat theta <= 0 as no theta, as the caller wants to ignore it.
    if theta is not None and theta <= 0:
        theta = None

    if theta is None and num_iterations is None:
        raise ValueError('Either theta or num_iterations (or both) must be provided.')

    if initial_V_S is None:
        V_S = np.array([0.0] * len(agent.SS))
    else:

        V_S = np.array([
            initial_V_S[s]
            for s in sorted(initial_V_S, key=lambda s: s.i)
        ])

        expected_shape = (len(agent.SS), )
        if V_S.shape != expected_shape:
            raise ValueError(
                f'Expected initial_V_S to have shape {expected_shape}, but it has shape {V_S.shape}')

    delta = None
    iterations_finished = 0
    while True:

        if update_in_place:
            V_S_to_update = V_S
        else:
            V_S_to_update = np.zeros_like(V_S)

        delta = 0.0

        for s_i, s in enumerate(agent.SS):

            prev_v = V_S[s_i]

            # calculate expected value of current state using current estimates of successor state-values
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

        if iterations_finished % 10 == 0:
            print(f'Finished {iterations_finished} iterations:  delta={delta}')

        # check termination conditions
        below_theta = theta is not None and delta < theta
        completed_num_iterations = num_iterations is not None and iterations_finished >= num_iterations
        if below_theta or completed_num_iterations:
            break

    print(f'Evaluation completed (iterations={iterations_finished}, delta={delta}).')

    # round the state-value function to reflect specified accuracy
    if theta is None:
        round_places = None
    else:
        round_places = int(abs(math.log10(theta)) - 1)

    return {
        s: v if round_places is None else round(v, round_places)
        for s, v in zip(agent.SS, V_S)
    }


@rl_text(chapter=4, page=76)
def evaluate_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: float,
        update_in_place: bool,
        initial_Q_S_A: Dict[MdpState, Dict[Action, float]] = None
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Minimum tolerated change in state-action value estimates, below which evaluation terminates.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_Q_S_A: Initial guess at state-action value, or None for no guess.
    :return: Dictionary of MDP states, actions, and their estimated values.
    """

    if theta <= 0:
        raise ValueError('theta must be > 0.0')

    if initial_Q_S_A is None:
        Q_S_A = {
            s: np.array([0.0] * len(agent.AA))
            for s in agent.SS
        }
    else:
        Q_S_A = {
            s: np.array([
                initial_Q_S_A[s][a]
                for a in sorted(initial_Q_S_A[s], key=lambda a: a.i)
            ])
            for s in initial_Q_S_A
        }

    delta = None
    iterations_finished = 0
    while delta is None or delta >= theta:

        if iterations_finished > 0 and iterations_finished % 10 == 0:
            print(f'Finished {iterations_finished} iterations:  delta={delta}')

        if update_in_place:
            Q_S_A_to_update = Q_S_A
        else:
            Q_S_A_to_update = {
                s: np.zeros_like(Q_S_A[s])
                for s in agent.SS
            }

        delta = 0.0

        # update each state-action value
        for s in agent.SS:
            for a_i, a in enumerate(agent.AA):

                prev_q = Q_S_A[s][a_i]

                # calculate expected state-action value using current estimates of successor state-action values
                new_q = np.sum([

                    # action is given, so start expectation with state/reward probability.
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
