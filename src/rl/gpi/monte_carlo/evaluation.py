from typing import Dict, Optional

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
    """
    Perform Monte Carlo policy evaluation of an agent's policy within an environment, returning state-action values.

    :param agent:
    :param environment:
    :param num_episodes: Number of episodes to execute.
    :return: Dictionary of MDP states and their estimated values under the agent's policy.
    """

    print(f'Running Monte Carlo evaluation of v_pi for {num_episodes} episode(s).')

    v_pi = {
        s: IncrementalSampleAverager()
        for s in environment.SS
    }

    episodes_per_print = int(num_episodes * 0.05)
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

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of G at each time step t to be properly discounted.
        G = 0
        for t, state, reward in reversed(t_state_reward):

            G = agent.gamma * G + reward.r

            # if this time step was the first visit to the state, then G is the sample value. add it to our average.
            if state_first_t[state] == t:
                v_pi[state].update(G)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    v_pi = {
        s: v_pi[s].get_value()
        for s in v_pi
    }

    return v_pi
