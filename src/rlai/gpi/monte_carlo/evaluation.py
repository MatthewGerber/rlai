from typing import Dict, Tuple, Set

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.gpi.utils import lazy_initialize_q_S_A, initialize_q_S_A
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager, sample_list_item


@rl_text(chapter=5, page=92)
def evaluate_v_pi(
        agent: MdpAgent,
        environment: MdpEnvironment,
        num_episodes: int
) -> Dict[MdpState, float]:
    """
    Perform Monte Carlo evaluation of an agent's policy within an environment, returning state values. Uses a random
    action on the first time step to maintain exploration (exploring starts). This evaluation approach is only
    marginally useful in practice, as the state-value estimates require a model of the environmental dynamics (i.e.,
    the transition-reward probability distribution) in order to be applied. See `evaluate_q_pi` in this module for a
    more feature-rich and useful evaluation approach (i.e., state-action value estimation). This evaluation function
    operates over rewards obtained at the end of episodes, so it is only appropriate for episodic tasks.

    :param agent: Agent.
    :param environment: Environment.
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
        state = environment.reset_for_new_run()
        agent.reset_for_new_run(state)

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

            next_state, next_t, reward = state.advance(environment, t, a)
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
        exploring_starts: bool,
        update_upon_every_visit: bool,
        off_policy_agent: MdpAgent = None,
        initial_q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]] = None
) -> Tuple[Dict[MdpState, Dict[Action, IncrementalSampleAverager]], Set[MdpState], float]:
    """
    Perform Monte Carlo evaluation of an agent's policy within an environment, returning state-action values. This
    evaluation function operates over rewards obtained at the end of episodes, so it is only appropriate for episodic
    tasks.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param exploring_starts: Whether or not to use exploring starts, forcing a random action in the first time step.
    This maintains exploration in the first state; however, unless each state has some nonzero probability of being
    selected as the first state, there is no assurance that all state-action pairs will be sampled. If the initial state
    is deterministic, consider passing False here and shifting the burden of exploration to the improvement step with
    a nonzero epsilon (see `rlai.gpi.improvement.improve_policy_with_q_pi`).
    :param update_upon_every_visit: True to update each state-action pair upon each visit within an episode, or False to
    update each state-action pair upon the first visit within an episode.
    :param off_policy_agent: Agent containing behavioral policy used to generate learning episodes. To ensure that the
    state-action value estimates converge to those of the target policy, the policy of the `off_policy_agent` must be
    soft (i.e., have positive probability for all state-action pairs that have positive probabilities in the agent's
    target policy).
    :param initial_q_S_A: Initial guess at state-action value, or None for no guess.
    :return: 3-tuple of (1) dictionary of all MDP states and their action-value averagers under the agent's policy, (2)
    set of only those states that were evaluated, and (3) the average reward obtained per episode.
    """

    print(f'Running Monte Carlo evaluation of q_pi for {num_episodes} episode(s).')

    evaluated_states = set()

    q_S_A = initialize_q_S_A(initial_q_S_A, environment, evaluated_states)

    episode_generation_agent = agent if off_policy_agent is None else off_policy_agent
    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    for episode_i in range(num_episodes):

        # reset the environment for the new run, and reset the agent accordingly.
        state = environment.reset_for_new_run()
        episode_generation_agent.reset_for_new_run(state)

        # simulate until episode termination, keeping a trace of state-action pairs and their immediate rewards, as well
        # as the times of their first visits (only if we're doing first-visit evaluation).
        t = 0
        state_action_first_t = None if update_upon_every_visit else {}
        t_state_action_reward = []
        total_reward = 0.0
        while not state.terminal:

            evaluated_states.add(state)

            if exploring_starts and t == 0:
                a = sample_list_item(state.AA, None, environment.random_state)
            else:
                a = episode_generation_agent.act(t)

            state_a = (state, a)

            # mark time step of first visit, if we're doing first-visit evaluation.
            if state_action_first_t is not None and state_a not in state_action_first_t:
                state_action_first_t[state_a] = t

            next_state, next_t, next_reward = state.advance(environment, t, a)
            t_state_action_reward.append((t, state_a, next_reward))
            total_reward += next_reward.r
            state = next_state
            t = next_t

            episode_generation_agent.sense(state, t)

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of G at each time step t to be properly discounted. here, W is the importance-sampling weight of the agent's
        # (target) policy compared to the episode generation policy (behavior).
        G = 0
        W = 1
        for t, state_a, reward in reversed(t_state_action_reward):

            G = agent.gamma * G + reward.r

            # if we're doing every-visit, or if the current time step was the first visit to the state-action, then G
            # is the discounted sample value. add it to our average.
            if state_action_first_t is None or state_action_first_t[state_a] == t:

                state, a = state_a

                lazy_initialize_q_S_A(q_S_A=q_S_A, state=state, a=a, alpha=None, weighted=True)

                # the following two lines work correctly for on- and off-policy learning. in the former case, the agent
                # and episode policies are the same, which makes W always equal to 1 (i.e., q_S_A is unweighted...the
                # on-policy case). in off-policy learning, W will be the importance-sampling weight.
                q_S_A[state][a].update(value=G, weight=W)
                W *= agent.pi[state][a] / episode_generation_agent.pi[state][a]

                # if the importance sampling weight becomes zero (allowing floating-point tolerane), then we're done,
                # as all subsequent weighted updates (at earlier time steps) will be zero. this is the sense in which
                # off-policy learning only learns from the "tails" of episodes in which all state-action pairs of the
                # episode are also greedy with respect to the agent's policy.
                if W < 0.00000001:
                    break

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            print(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    print(f'Completed evaluation. Average reward per episode:  {episode_reward_averager.get_value()}')

    return q_S_A, evaluated_states, episode_reward_averager.get_value()
