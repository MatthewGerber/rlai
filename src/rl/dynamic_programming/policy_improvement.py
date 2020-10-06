from typing import Dict

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.environments.mdp import MdpEnvironment
from rl.meta import rl_text
from rl.states.mdp import MdpState


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
