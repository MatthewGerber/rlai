import logging
from typing import Dict, Tuple, Set, Optional

import numpy as np

from rlai.core import MdpState
from rlai.core.environments.mdp import MdpEnvironment
from rlai.docs import rl_text
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.utils import IncrementalSampleAverager, sample_list_item


@rl_text(chapter=5, page=92)
def evaluate_v_pi(
        agent: ActionValueMdpAgent,
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

    logging.info(f'Running Monte Carlo evaluation of v_pi for {num_episodes} episode(s).')

    v_pi: Dict[MdpState, IncrementalSampleAverager] = {
        terminal_state: IncrementalSampleAverager()
        for terminal_state in environment.terminal_states
    }

    episodes_per_print = int(num_episodes * 0.05)
    for episode_i in range(num_episodes):

        # start the environment in a random state
        state = environment.reset_for_new_run(agent)
        agent.reset_for_new_run(state)
        agent.q_S_A.reset_for_new_run(state)

        # simulate until episode termination, keeping a trace of states and their immediate rewards, as well as the
        # times of their first visits.
        t = 0
        state_first_t = {}
        t_state_reward = []
        truncation_time_step = None
        while not state.terminal:
            try:

                if state.truncated and truncation_time_step is None:
                    truncation_time_step = t
                    logging.info(f'Episode was truncated after {t} step(s).')
                elif not state.truncated and truncation_time_step is not None:
                    raise ValueError('Truncation cannot be exited.')

                if state not in state_first_t:
                    state_first_t[state] = t

                # force exploring starts. if this is not forced, then the agent's policy might be deterministic at the first
                # time step and might prevent exploration of all state-action sequences.
                if t == 0:
                    a = sample_list_item(state.AA, None, environment.random_state)
                else:
                    a = agent.act(t)

                next_state, next_reward = environment.advance(state, t, a, agent)
                t_state_reward.append((t, state, next_reward))
                state = next_state
                t += 1
                agent.sense(state, t)

                # if we've truncated and the discount has converged to zero, then the return at the truncation time will
                # not change by running longer. we've got an accurate return estimate at truncation. exit the episode.
                if truncation_time_step is not None:
                    num_post_truncation_steps = (t - truncation_time_step)
                    post_truncation_discount = agent.gamma ** num_post_truncation_steps
                    if np.isclose(post_truncation_discount, 0.0):
                        raise ValueError(
                            f'Post-truncation discount converged to zero after {num_post_truncation_steps} step(s).'
                        )

            # if anything blows up, then let the environment know that we are exiting the episode.
            except ValueError as e:
                logging.info(f'{e} Exiting episode without termination.')
                environment.exiting_episode_without_termination()
                break

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of g at each time step t to be properly discounted.
        g = 0.0
        for t, state, reward in reversed(t_state_reward):

            g = agent.gamma * g + reward.r

            # only update value estimates before the truncation time step if we have one
            if truncation_time_step is not None and t >= truncation_time_step:
                continue

            # if the current time step was the first visit to the state, then g is the discounted sample value. add it
            # to our average.
            if state_first_t[state] == t:
                if state in v_pi:
                    value_estimator = v_pi[state]
                else:
                    value_estimator = IncrementalSampleAverager()
                    v_pi[state] = value_estimator
                value_estimator.update(g)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            logging.info(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    return {
        s: v_pi[s].get_value()
        for s in v_pi
    }


@rl_text(chapter=5, page=96)
def evaluate_q_pi(
        agent: ActionValueMdpAgent,
        environment: MdpEnvironment,
        num_episodes: int,
        exploring_starts: bool,
        update_upon_every_visit: bool,
        off_policy_agent: Optional[ActionValueMdpAgent] = None
) -> Tuple[Set[MdpState], float]:
    """
    Perform Monte Carlo evaluation of an agent's policy within an environment. This evaluation function operates over
    rewards obtained at the end of episodes, so it is only appropriate for episodic tasks.

    :param agent: Agent containing target policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param exploring_starts: Whether to use exploring starts, forcing a random action in the first time step.
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
    :return: 2-tuple of (1) set of only those states that were evaluated, and (2) the average reward obtained per
    episode.
    """

    logging.info(f'Running Monte Carlo evaluation of q_pi for {num_episodes} episode(s).')

    evaluated_states = set()

    episode_generation_agent = agent if off_policy_agent is None else off_policy_agent
    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    for episode_i in range(num_episodes):

        # reset the environment for the new run (always use the agent we're learning about, as state identifiers come
        # from it), and reset the episode generate agent accordingly.
        state = environment.reset_for_new_run(agent)
        episode_generation_agent.reset_for_new_run(state)

        # simulate until episode termination, keeping a trace of state-action pairs and their immediate rewards, as well
        # as the times of their first visits (only if we're doing first-visit evaluation).
        t = 0
        state_action_first_t: Optional[Dict] = None if update_upon_every_visit else {}
        t_state_action_reward = []
        total_reward = 0.0
        truncation_time_step = None
        while not state.terminal:
            try:

                # mark truncation time and exclude the state from those that were properly evaluated
                if state.truncated:
                    if truncation_time_step is None:
                        truncation_time_step = t
                        logging.info(f'Episode was truncated after {t} step(s).')
                else:
                    evaluated_states.add(state)

                if not state.truncated and truncation_time_step is not None:
                    raise ValueError('Truncation cannot be exited.')

                if exploring_starts and t == 0:
                    a = sample_list_item(state.AA, None, environment.random_state)
                else:
                    a = episode_generation_agent.act(t)

                state_a = (state, a)

                # mark time step of first visit, if we're doing first-visit evaluation.
                if state_action_first_t is not None and state_a not in state_action_first_t:
                    state_action_first_t[state_a] = t

                next_state, next_reward = environment.advance(state, t, a, agent)
                t_state_action_reward.append((t, state_a, next_reward))
                total_reward += next_reward.r
                state = next_state
                t += 1
                episode_generation_agent.sense(state, t)

                # if we've truncated and the discount has converged to zero, then the return at the truncation time will
                # not change by running longer. we've got an accurate return estimate at truncation. exit the episode.
                if truncation_time_step is not None:
                    num_post_truncation_steps = (t - truncation_time_step)
                    post_truncation_discount = agent.gamma ** num_post_truncation_steps
                    if np.isclose(post_truncation_discount, 0.0):
                        raise ValueError(
                            f'Post-truncation discount converged to zero after {num_post_truncation_steps} step(s).'
                        )

            # if anything blows up, then let the environment know that we are exiting the episode.
            except ValueError as e:
                logging.info(f'{e} Exiting episode without termination.')
                environment.exiting_episode_without_termination()
                break

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of g at each time step t to be properly discounted. here, w is the importance-sampling weight of the agent's
        # (target) policy compared to the episode generation policy (behavior).
        g = 0.0
        w = 1.0
        for t, state_a, reward in reversed(t_state_action_reward):

            g = agent.gamma * g + reward.r

            # only update value estimates before the truncation time step if we have one
            if truncation_time_step is not None and t >= truncation_time_step:
                continue

            # if we're doing every-visit, or if the current time step was the first visit to the state-action, then g
            # is the discounted sample value. add it to our average.
            if state_action_first_t is None or state_action_first_t[state_a] == t:

                state, a = state_a

                agent.q_S_A.initialize(state=state, a=a, alpha=None, weighted=True)

                # the following two lines work correctly for on- and off-policy learning. in the former case, the agent
                # and episode policies are the same, which makes w always equal to 1 (i.e., q_S_A is unweighted...the
                # on-policy case). in off-policy learning, w will be the importance-sampling weight.
                agent.q_S_A[state][a].update(value=g, weight=w)
                w *= agent.pi[state][a] / episode_generation_agent.pi[state][a]

                # if the importance sampling weight becomes zero (allowing floating-point tolerance), then we're done,
                # as all subsequent weighted updates (at earlier time steps) will be zero. this is the sense in which
                # off-policy learning only learns from the "tails" of episodes in which all state-action pairs of the
                # episode are also greedy with respect to the agent's policy.
                if w < 0.00000001:
                    break

        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            logging.info(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    logging.info(f'Completed evaluation. Average reward per episode:  {episode_reward_averager.get_value()}')

    return evaluated_states, episode_reward_averager.get_value()
