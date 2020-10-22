from typing import Dict, Tuple

from rl.actions import Action
from rl.agents.mdp import MdpAgent
from rl.environments.mdp import MdpEnvironment
from rl.meta import rl_text
from rl.states.mdp import MdpState
from rl.utils import IncrementalSampleAverager, sample_list_item


@rl_text(chapter=5, page=92)
def evaluate_v_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_episodes: int
) -> Dict[MdpState, float]:
    """
    Perform Monte Carlo evaluation of an agent's policy within an environment, returning state values. Uses a random
    action on the first time step to maintain exploration (exploring starts).

    :param agent:
    :param environment:
    :param num_episodes: Number of episodes to execute.
    :return: Dictionary of MDP states and their estimated values under the agent's policy.
    """

    print(f'Running Monte Carlo evaluation of v_pi for {num_episodes} episode(s).')

    v_pi: Dict[MdpState, IncrementalSampleAverager] = {
        terminal_state: IncrementalSampleAverager()
        for terminal_state in environment.terminal_states
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

            if t == 0:
                a = sample_list_item(state.AA, None, environment.random_state)
            else:
                a = agent.act(t)

            next_state, next_t, reward = state.advance(a, t, environment.random_state)
            t_state_reward.append((t, state, reward))
            state = next_state
            t = next_t

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of G at each time step t to be properly discounted.
        G = 0
        for t, state, reward in reversed(t_state_reward):

            G = agent.gamma * G + reward.r

            # if the current time step was the first visit to the state, then G is the discounted sample value. add it
            # to our average.
            if state_first_t[state] == t:

                if state not in v_pi:
                    v_pi[state] = IncrementalSampleAverager()

                v_pi[state].update(G)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    return {
        s: v_pi[s].get_value()
        for s in v_pi
    }


@rl_text(chapter=5, page=96)
def evaluate_q_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_episodes: int,
        initial_q_S_A: Dict[MdpState, Dict[Action, float]] = None
) -> Tuple[Dict[MdpState, Dict[Action, float]], float]:
    """
    Perform Monte Carlo evaluation of an agent's policy within an environment, returning state-action values. Uses a
    random action on the first time step to maintain exploration (exploring starts).

    :param agent: Agent.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 2-tuple of (1) dictionary of MDP states and their estimated values under the agent's policy, and (2) the
    per-episode average reward obtained.
    """

    print(f'Running Monte Carlo evaluation of q_pi for {num_episodes} episode(s).')

    if initial_q_S_A is None:

        # start with an averager for each terminal state, which should never be updated.
        q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]] = {
            terminal_state: {
                a: IncrementalSampleAverager()
                for a in terminal_state.AA
            }
            for terminal_state in environment.terminal_states
        }

    else:

        # set initial guesses
        q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]] = {
            s: {
                a: IncrementalSampleAverager(initial_value=initial_q_S_A[s][a])
                for a in initial_q_S_A[s]
            }
            for s in initial_q_S_A
        }

    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    for episode_i in range(num_episodes):

        # start the environment in a random state with a random feasible action in that state
        environment.reset_for_new_run(agent)
        state = environment.state

        # simulate until episode termination, keeping a trace of state-action pairs and their immediate rewards, as well
        # as the times of their first visits.
        t = 0
        state_action_first_t = {}
        t_state_action_reward = []
        total_reward = 0.0
        while not state.terminal:

            if t == 0:
                a = sample_list_item(state.AA, None, environment.random_state)
            else:
                a = agent.act(t)

            state_a = (state, a)

            if state_a not in state_action_first_t:
                state_action_first_t[state_a] = t

            next_state, next_t, reward = state.advance(a, t, environment.random_state)
            t_state_action_reward.append((t, state_a, reward))
            total_reward += reward.r
            state = next_state
            t = next_t

            agent.sense(state, t)

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of G at each time step t to be properly discounted.
        G = 0
        for t, state_a, reward in reversed(t_state_action_reward):

            G = agent.gamma * G + reward.r

            # if the current time step was the first visit to the state-action, then G is the discounted sample value.
            # add it to our average.
            if state_action_first_t[state_a] == t:

                state, a = state_a

                if state not in q_S_A:
                    q_S_A[state] = {}

                if a not in q_S_A[state]:
                    q_S_A[state][a] = IncrementalSampleAverager()

                q_S_A[state][a].update(G)

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    print(f'Completed evaluation. Average reward per episode:  {episode_reward_averager.get_value()}')

    q_pi = {
        s: {
            a: q_S_A[s][a].get_value()
            for a in q_S_A[s]
        }
        for s in q_S_A
    }

    return q_pi, episode_reward_averager.get_value()
