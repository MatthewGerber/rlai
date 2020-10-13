import math
from typing import Optional, Dict, Tuple

import numpy as np

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.meta import rl_text
from rl.states.mdp import MdpState


@rl_text(chapter=4, page=74)
def evaluate_v_pi(
        agent: MdpAgent,
        theta: Optional[float],
        num_iterations: Optional[int],
        update_in_place: bool,
        initial_v_S: Optional[Dict[MdpState, float]] = None
) -> Tuple[Dict[MdpState, float], float]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state values.

    :param agent: MDP agent.
    :param theta: Minimum tolerated change in state value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_v_S: Initial guess at state-value, or None for no guess.
    :return: 2-tuple of (1) dictionary of MDP states and their estimated values under the agent's policy, and (2) final
    value of delta.
    """

    theta, num_iterations = check_termination_criteria(
        theta=theta,
        num_iterations=num_iterations
    )

    if initial_v_S is None:
        v_S = {s: 0.0 for s in agent.SS}
    else:
        v_S = initial_v_S

    iterations_finished = 0
    while True:

        if update_in_place:
            v_S_to_update = v_S
        else:
            v_S_to_update = {s: 0.0 for s in agent.SS}

        delta = 0.0

        for s in agent.SS:

            prev_v = v_S[s]

            # calculate expected value of current state using current estimates of successor state-values
            new_v = np.sum([

                agent.pi[s][a] * s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * v_S[s_prime])

                for a in s.p_S_prime_R_given_A
                for s_prime in s.p_S_prime_R_given_A[a]
                for r in s.p_S_prime_R_given_A[a][s_prime]
            ])

            v_S_to_update[s] = new_v

            delta = max(delta, abs(prev_v - new_v))

        if not update_in_place:
            v_S = v_S_to_update

        iterations_finished += 1

        if check_termination_conditions(
            delta=delta,
            theta=theta,
            iterations_finished=iterations_finished,
            num_iterations=num_iterations
        ):
            break

    v_pi = {
        s: round_for_theta(v, theta)
        for s, v in v_S.items()
    }

    return v_pi, delta


@rl_text(chapter=4, page=76)
def evaluate_q_pi(
        agent: MdpAgent,
        theta: Optional[float],
        num_iterations: Optional[int],
        update_in_place: bool,
        initial_q_S_A: Dict[MdpState, Dict[Action, float]] = None
) -> Tuple[Dict[MdpState, Dict[Action, float]], float]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: MDP agent.
    :param theta: Minimum tolerated change in state value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 2-tuple of (1) dictionary of MDP states, actions, and their estimated values under the agent's policy, and
    (2) final value of delta.
    """

    theta, num_iterations = check_termination_criteria(
        theta=theta,
        num_iterations=num_iterations
    )

    if initial_q_S_A is None:
        q_S_A = {
            s: {
                a: 0.0
                for a in s.p_S_prime_R_given_A
            }
            for s in agent.SS
        }
    else:
        q_S_A = initial_q_S_A

    iterations_finished = 0
    while True:

        if update_in_place:
            q_S_A_to_update = q_S_A
        else:
            q_S_A_to_update = {
                s: {
                    a: 0.0
                    for a in s.p_S_prime_R_given_A
                }
                for s in agent.SS
            }

        delta = 0.0

        # update each state-action value
        for s in agent.SS:
            for a in s.p_S_prime_R_given_A:

                prev_q = q_S_A[s][a]

                # calculate expected state-action value using current estimates of successor state-action values
                new_q = np.sum([

                    # action is given, so start expectation with state/reward probability.
                    s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * np.sum([
                        agent.pi[s_prime][a_prime] * q_S_A[s_prime][a_prime]
                        for a_prime in s.p_S_prime_R_given_A
                     ]))

                    for s_prime in s.p_S_prime_R_given_A[a]
                    for r in s.p_S_prime_R_given_A[a][s_prime]
                ])

                q_S_A_to_update[s][a] = new_q

                delta = max(delta, abs(prev_q - new_q))

        if not update_in_place:
            q_S_A = q_S_A_to_update

        iterations_finished += 1

        if check_termination_conditions(
            delta=delta,
            theta=theta,
            iterations_finished=iterations_finished,
            num_iterations=num_iterations
        ):
            break

    q_pi = {
        s: {
            a: round_for_theta(q, theta)
            for a, q in q_S_A[s].items()
        }
        for s in q_S_A
    }

    return q_pi, delta


def check_termination_criteria(
        theta: Optional[float],
        num_iterations: Optional[int]
) -> Tuple[float, int]:
    """
    Check theta and number of iterations.

    :param theta: Theta.
    :param num_iterations: Number of iterations.
    :return: Normalized values.
    """

    # treat theta <= 0 as None, as the caller wants to ignore it.
    if theta is not None and theta <= 0:
        theta = None

    # treat num_iterations <= 0 as None, as the caller wants to ignore it.
    if num_iterations is not None and num_iterations <= 0:
        num_iterations = None

    if theta is None and num_iterations is None:
        raise ValueError('Either theta or num_iterations (or both) must be provided.')

    print(f'Starting evaluation:  theta={theta}, num_iterations={num_iterations}')

    return theta, num_iterations


def check_termination_conditions(
        delta: float,
        theta: Optional[float],
        iterations_finished: int,
        num_iterations: Optional[int]
) -> bool:
    """
    Check for termination.

    :param delta: Delta.
    :param theta: Theta.
    :param iterations_finished: Number of iterations that have been finished.
    :param num_iterations: Maximum number of iterations.
    :return: True for termination.
    """

    if iterations_finished % 10 == 0:
        print(f'\tFinished {iterations_finished} iterations:  delta={delta}')

    below_theta = theta is not None and delta < theta
    completed_num_iterations = num_iterations is not None and iterations_finished >= num_iterations

    if below_theta or completed_num_iterations:
        print(f'\tEvaluation completed:  iterations={iterations_finished}, delta={delta}\n')
        return True
    else:
        return False


def round_for_theta(
        v: float,
        theta: Optional[float]
) -> float:
    """
    Round a value based on the precision of theta.

    :param v: Value.
    :param theta: Theta.
    :return: Rounded value.
    """

    if theta is None:
        return v
    else:
        return round(v, int(abs(math.log10(theta)) - 1))
