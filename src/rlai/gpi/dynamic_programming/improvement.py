from typing import Dict

from rlai.agents.mdp import MdpAgent
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.meta import rl_text
from rlai.states.mdp import ModelBasedMdpState


@rl_text(chapter=4, page=76)
def improve_policy_with_v_pi(
        agent: MdpAgent,
        v_pi: Dict[ModelBasedMdpState, float]
) -> int:
    """
    Improve an agent's policy according to its state-value estimates. This makes the policy greedy with respect to the
    state-value estimates. In cases where multiple such greedy actions exist for a state, each of the greedy actions
    will be assigned equal probability.

    Note that the present function resides within `rlai.gpi.dynamic_programming.improvement` and requires state-value
    estimates of states that are model-based. These are the case because policy improvement from state values is only
    possible if we have a model of the environment. Compare with `rlai.gpi.improvement.improve_policy_with_q_pi`, which
    accepts model-free states since state-action values are estimated directly.

    :param agent: Agent.
    :param v_pi: State-value estimates for the agent's policy.
    :return: Number of states in which the policy was updated.
    """

    # calculate state-action values (q) for the agent's policy
    s: ModelBasedMdpState
    q_S_A = {
        s: {
            a: sum([
                s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * v_pi[s_prime])
                for s_prime in s.p_S_prime_R_given_A[a]
                for r in s.p_S_prime_R_given_A[a][s_prime]
            ])
            for a in s.p_S_prime_R_given_A
        }
        for s in agent.pi
    }

    return improve_policy_with_q_pi(
        agent=agent,
        q_pi=q_S_A
    )
