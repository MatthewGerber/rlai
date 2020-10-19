from typing import Dict

from rl.agents.mdp import MdpAgent
from rl.environments.mdp import MdpEnvironment
from rl.meta import rl_text
from rl.states.mdp import MdpState
from rl.utils import IncrementalSampleAverager


@rl_text(chapter=5, page=92)
def evaluate_v_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_episodes: int
) -> Dict[MdpState, float]:

    v_pi = {
        s: IncrementalSampleAverager()
        for s in environment.SS
    }

    for episode_i in range(num_episodes):

        # start the environment in a random state
        environment.reset_for_new_run(agent)
        state = environment.state

        # simulate until episode termination, keeping a trace of states and their immediate rewards, as well as the
        # times of their first visits.
        t = 0
        state_first_t = {}
        t_state_reward = []
        while not state.terminal:

            if state not in state_first_t:
                state_first_t[state] = t

            a = agent.act(t)
            next_state, reward = state.advance(a, environment.random_state)
            t_state_reward.append((t, state, reward))
            state = next_state
            t += 1

        # work backwards through the trace to calculate discounted returns
        G = 0
        for t, state, reward in reversed(t_state_reward):
            G = agent.gamma * G + reward.r

            # add discounted return to sample estimate, if this was the first visit.
            if state_first_t[state] == t:
                v_pi[state].update(G)

    return {
        s: v_pi[s].get_value()
        for s in v_pi
    }
