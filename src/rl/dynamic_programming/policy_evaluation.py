import math
from typing import Optional, Dict, Tuple

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
        initial_v_S: Optional[Dict[MdpState, float]] = None
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
    :param initial_v_S: Initial guess at state-value, or None for no guess.
    :return: Dictionary of MDP states and their estimated values.
    """

    theta, num_iterations = check_termination_criteria(
        theta=theta,
        num_iterations=num_iterations
    )

    if initial_v_S is None:
        v_S = np.array([0.0] * len(agent.SS))
    else:

        v_S = np.array([
            initial_v_S[s]
            for s in sorted(initial_v_S, key=lambda s: s.i)
        ])

        expected_shape = (len(agent.SS), )
        if v_S.shape != expected_shape:
            raise ValueError(
                f'Expected initial_v_S to have shape {expected_shape}, but it has shape {v_S.shape}')

    iterations_finished = 0
    while True:

        if update_in_place:
            v_S_to_update = v_S
        else:
            v_S_to_update = np.zeros_like(v_S)

        delta = 0.0

        for s_i, s in enumerate(agent.SS):

            prev_v = v_S[s_i]

            # calculate expected value of current state using current estimates of successor state-values
            new_v = np.sum([

                agent.pi[s][a] * s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * v_S[s_prime_i])

                for a in agent.AA
                for s_prime_i, s_prime in enumerate(agent.SS)
                for r in environment.RR
            ])

            v_S_to_update[s_i] = new_v

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

    return {
        s: round_for_theta(v, theta)
        for s, v in zip(agent.SS, v_S)
    }


@rl_text(chapter=4, page=76)
def evaluate_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        theta: Optional[float],
        num_iterations: Optional[int],
        update_in_place: bool,
        initial_q_S_A: Dict[MdpState, Dict[Action, float]] = None
) -> Dict[MdpState, Dict[Action, float]]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Minimum tolerated change in state value estimates, below which evaluation terminates. Either `theta`
    or `num_iterations` (or both) can be specified, but passing neither will raise an exception.
    :param num_iterations: Number of evaluation iterations to execute.  Either `theta` or `num_iterations` (or both)
    can be specified, but passing neither will raise an exception.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: Dictionary of MDP states, actions, and their estimated values.
    """

    theta, num_iterations = check_termination_criteria(
        theta=theta,
        num_iterations=num_iterations
    )

    if initial_q_S_A is None:
        q_S_A = {
            s: np.array([0.0] * len(agent.AA))
            for s in agent.SS
        }
    else:
        q_S_A = {
            s: np.array([
                initial_q_S_A[s][a]
                for a in sorted(initial_q_S_A[s], key=lambda a: a.i)
            ])
            for s in initial_q_S_A
        }

    iterations_finished = 0
    while True:

        if update_in_place:
            q_S_A_to_update = q_S_A
        else:
            q_S_A_to_update = {
                s: np.zeros_like(q_S_A[s])
                for s in agent.SS
            }

        delta = 0.0

        # update each state-action value
        for s in agent.SS:
            for a_i, a in enumerate(agent.AA):

                prev_q = q_S_A[s][a_i]

                # calculate expected state-action value using current estimates of successor state-action values
                new_q = np.sum([

                    # action is given, so start expectation with state/reward probability.
                    s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * np.sum([
                        agent.pi[s_prime][a_prime] * q_S_A[s_prime][a_prime_i]
                        for a_prime_i, a_prime in enumerate(agent.AA)
                     ]))

                    for s_prime_i, s_prime in enumerate(agent.SS)
                    for r in environment.RR
                ])

                q_S_A_to_update[s][a_i] = new_q

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

    return {
        s: {
            a: round_for_theta(q, theta)
            for a, q in zip(agent.AA, q_S_A[s])
        }
        for s in agent.SS
    }


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
