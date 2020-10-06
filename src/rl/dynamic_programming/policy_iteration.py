import math
from typing import Dict, Optional

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
        theta: float,
        update_in_place: bool,
        initial_V_S: Optional[Dict[MdpState, float]] = None
) -> Dict[MdpState, float]:
    """
    Perform iterative policy evaluation of an agent's policy within an environment, returning state values.

    :param agent: MDP agent.
    :param environment: MDP environment.
    :param theta: Prediction accuracy requirement.
    :param update_in_place: Whether or not to update value estimates in place.
    :param initial_V_S: Initial guess at state-value, or None for no guess.
    :return: Dictionary of MDP states and their estimated values.
    """

    if theta <= 0:
        raise ValueError('theta must be > 0.0')

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
    while delta is None or delta > theta:

        if iterations_finished > 0 and iterations_finished % 10 == 0:
            print(f'Finished {iterations_finished} iterations:  delta={delta}')

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

    print(f'Evaluation completed after {iterations_finished} iteration(s).')

    round_places = int(abs(math.log10(theta)) - 1)

    return {
        s: round(v, round_places)
        for s, v in zip(agent.SS, V_S)
    }


@rl_text(chapter=4, page=76)
def improve_policy_with_v_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        v_pi: Dict[MdpState, float]
) -> bool:
    """
    Improve an agent's policy according to its state-value estimates. This makes the policy greedy with respect to the
    state-value estimates. In cases where multiple such greedy actions exist for a state, each of the greedy actions
    will be assigned equal probability.

    :param agent: Agent.
    :param environment: Environment.
    :param v_pi: State-value estimates for the agent's policy.
    :return: True if policy was changed and False if the policy was not changed.
    """

    # calculate state-action values (q) for the agent's policy
    Q_S_A = {
        s: {
            a: sum([
                s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * v_pi[s_prime])
                for s_prime in agent.SS
                for r in environment.RR
            ])
            for a in agent.AA
        }
        for s in agent.SS
    }

    return improve_policy_with_q_pi(
        agent=agent,
        q_pi=Q_S_A
    )


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
    :param theta: See `evaluate_v`.
    :param update_in_place: See `evaluate_v`.
    :return: Final state-value estimates.
    """

    v_pi: Optional[Dict[MdpState, float]] = None
    improving = True
    i = 0
    while improving:

        v_pi = evaluate_v_pi(
            agent=agent,
            environment=environment,
            theta=theta,
            update_in_place=update_in_place,
            initial_V_S=v_pi
        )

        improving = improve_policy_with_v_pi(
            agent=agent,
            environment=environment,
            v_pi=v_pi
        )

        i += 1

    print(f'Policy iteration terminated after {i} iteration(s).')

    return v_pi


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
    :param theta: Prediction accuracy requirement.
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
    while delta is None or delta > theta:

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

                    # action is given, so start expectation with state-reward probability.
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


@rl_text(chapter=4, page=76)
def improve_policy_with_q_pi(
        agent: MdpAgent,
        q_pi: Dict[MdpState, Dict[Action, float]]
) -> bool:
    """
    Improve an agent's policy according to its state-action value estimates. This makes the policy greedy with respect
    to the state-action value estimates. In cases where multiple such greedy actions exist for a state, each of the
    greedy actions will be assigned equal probability.

    :param agent: Agent.
    :param q_pi: State-action value estimates for the agent's policy.
    :return: True if policy was changed and False if the policy was not changed.
    """

    # get the maximal action value for each state
    S_max_Q = {
        s: max(q_pi[s].values())
        for s in q_pi
    }

    # count up how many actions in each state are maximizers
    S_num_A_max_Q = {
        s: sum(q_pi[s][a] == S_max_Q[s] for a in q_pi[s])
        for s in q_pi
    }

    # update policy, assigning uniform probability across all maximizing actions.
    agent_old_pi = agent.pi
    agent.pi = {
        s: {
            a: 1.0 / S_num_A_max_Q[s] if q_pi[s][a] == S_max_Q[s] else 0.0
            for a in agent.AA
        }
        for s in agent.SS
    }

    # check our math
    if not all(
        sum(agent.pi[s].values()) == 1.0
        for s in agent.pi
    ):
        raise ValueError('Expected action probabilities to sum to 1.0')

    return agent_old_pi != agent.pi


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
    :param theta: See `evaluate_v_pi`.
    :param update_in_place: See `evaluate_v_pi`.
    :return: Final state-action value estimates.
    """

    q_pi: Optional[Dict[MdpState, Dict[Action, float]]] = None
    improving = True
    i = 0
    while improving:

        q_pi = evaluate_q_pi(
            agent=agent,
            environment=environment,
            theta=theta,
            update_in_place=update_in_place,
            initial_Q_S_A=q_pi
        )

        improving = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi
        )

        i += 1

    print(f'Policy iteration terminated after {i} iteration(s).')

    return q_pi
