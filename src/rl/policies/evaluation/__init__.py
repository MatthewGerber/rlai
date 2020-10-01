from typing import Dict

import numpy as np

from rl.agents.policy import PolicyAgent
from rl.environments.mdp import MDP
from rl.meta import rl_text
from rl.states import State
from rl.states.mdp import MdpState


@rl_text(chapter=4, page=75)
def evaluate_policy_state_values(
        agent: PolicyAgent,
        environment: MDP,
        theta: float
) -> Dict[State, float]:

    if theta <= 0:
        raise ValueError('Theta must be > 0.')

    V_S = np.array([0.0] * len(environment.SS))

    delta = theta + 1
    s: MdpState
    while delta > theta:

        prev_V_S = V_S.copy()

        V_S = np.array([
            np.sum([
                agent.pi[s][a] * s.p_S_prime_R_given_A[a][s_prime][r] * (r.r + agent.gamma * V_S[s_prime_i])
                for a in environment.AA
                for s_prime_i, s_prime in enumerate(environment.SS)
                for r in environment.RR
            ])
            for s in environment.SS
        ])

        delta = (prev_V_S - V_S).abs().max()

    return {
        s: v
        for s, v in zip(environment.SS, V_S)
    }
