from typing import Dict, Optional

import numpy as np

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.meta import rl_text
from rlai.states.mdp import MdpState


@rl_text(chapter=4, page=76)
def improve_policy_with_q_pi(
        agent: MdpAgent,
        q_pi: Dict[MdpState, Dict[Action, float]],
        epsilon: Optional[float] = None
) -> int:
    """
    Improve an agent's policy according to its state-action value estimates. This makes the policy greedy with respect
    to the state-action value estimates. In cases where multiple such greedy actions exist for a state, each of the
    greedy actions will be assigned equal probability.

    :param agent: Agent.
    :param q_pi: State-action value estimates for the agent's policy.
    :param epsilon: Total probability mass to spread across all actions, resulting in an epsilon-greedy policy. Must
    be >= 0 if provided.
    :return: Number of states in which the policy was updated.
    """

    if epsilon is None:
        epsilon = 0.0
    elif epsilon < 0.0:
        raise ValueError('Epsilon must be >= 0')

    # get the maximal action value for each state
    S_max_q = {
        s: max(q_pi[s].values())
        for s in q_pi
    }

    # count up how many actions in each state are maximizers (i.e., tied in action value)
    S_num_maximizers = {
        s: sum(q_pi[s][a] == S_max_q[s] for a in q_pi[s])
        for s in q_pi
    }

    # generate policy update, assigning uniform probability across all maximizing actions in addition to a uniform
    # fraction of epsilon spread across all actions in the state.
    policy_update = {
        s: {
            a:
                ((1.0 - epsilon) / S_num_maximizers[s]) + (epsilon / len(s.AA)) if a in q_pi[s] and q_pi[s][a] == S_max_q[s]
                else epsilon / len(s.AA)

            # update policy for all feasible actions in the state
            for a in s.AA
        }
        for s in agent.pi

        # we can only update the policy for states that we have q-value estimates for
        if s in q_pi
    }

    # count up how many states got a new action distribution
    num_states_updated = sum(
        any(
            agent.pi[s][a] != policy_update[s][a]
            for a in policy_update[s]
        )
        for s in policy_update
    )

    # execute update on policy
    agent.pi.update(policy_update)

    # check that the action probabilities in each state sum to 1.0
    if not np.allclose(
        [
            sum(agent.pi[s].values())
            for s in agent.pi
        ], 1.0
    ):
        raise ValueError('Expected action probabilities to sum to 1.0')

    return num_states_updated
