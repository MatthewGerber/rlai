import logging
import os
import pickle
import tempfile
import warnings
from typing import Optional

import numpy as np

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies.parameterized import ParameterizedPolicy
from rlai.policies.parameterized.continuous_action import ContinuousActionPolicy
from rlai.utils import (
    IncrementalSampleAverager,
    RunThreadManager,
    ScatterPlot,
    insert_index_into_path
)
from rlai.v_S import StateValueEstimator


@rl_text(chapter=13, page=326)
def improve(
        agent: StochasticMdpAgent,
        policy: ParameterizedPolicy,
        environment: MdpEnvironment,
        num_episodes: int,
        update_upon_every_visit: bool,
        alpha: float,
        v_S: Optional[StateValueEstimator],
        thread_manager: RunThreadManager,
        plot_state_value: bool,
        num_episodes_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        training_pool_directory: Optional[str] = None,
        training_pool_batch_size: Optional[int] = None
) -> Optional[str]:
    """
    Perform Monte Carlo improvement of an agent's policy within an environment via the REINFORCE policy gradient method.
    This improvement function operates over rewards obtained at the end of episodes, so it is only appropriate for
    episodic tasks.

    :param agent: Agent containing target policy to be optimized.
    :param policy: Parameterized policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param update_upon_every_visit: True to update each state-action pair upon each visit within an episode, or False to
    update each state-action pair upon the first visit within an episode.
    :param alpha: Policy gradient step size.
    :param thread_manager: Thread manager. The current function (and the thread running it) will wait on this manager
    before starting each iteration. This provides a mechanism for pausing, resuming, and aborting training. Omit for no
    waiting.
    :param plot_state_value: Whether or not to plot the state-value.
    :param v_S: Baseline state-value estimator, or None for no baseline.
    :param num_episodes_per_checkpoint: Number of episodes per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_episodes_per_checkpoint` is provided.
    :param training_pool_directory: Path to directory in which to store pooled training runs.
    :param training_pool_batch_size: Number of episodes per training pool batch.
    :return: Final checkpoint path, or None if checkpoints were not saved.
    """

    if thread_manager is not None:
        warnings.warn('This optimization method will ignore the thread_manager.')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    # prepare training pool
    training_pool_path = None
    has_training_pool_directory = training_pool_directory is not None
    has_training_pool_batch_size = training_pool_batch_size is not None
    if has_training_pool_directory != has_training_pool_batch_size:
        raise ValueError('Both training pool directory and batch size must be provided, or neither.')
    elif has_training_pool_directory:
        training_pool_directory = os.path.expanduser(training_pool_directory)
        if not os.path.exists(training_pool_directory):
            os.mkdir(training_pool_directory)

        training_pool_path = tempfile.NamedTemporaryFile(dir=training_pool_directory, delete=False).name

    state_value_plot = None
    if plot_state_value and v_S is not None:
        state_value_plot = ScatterPlot('REINFORCE:  State Value', ['Estimate'], None)

    logging.info(f'Running Monte Carlo-based REINFORCE improvement for {num_episodes} episode(s).')

    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    final_checkpoint_path = None
    for episode_i in range(num_episodes):

        # reset the environment for the new run (always use the agent we're learning about, as state identifiers come
        # from it), and reset the agent accordingly.
        state = environment.reset_for_new_run(agent)
        agent.reset_for_new_run(state)

        # simulate until episode termination, keeping a trace of state-action pairs and their immediate rewards, as well
        # as the times of their first visits (only if we're doing first-visit evaluation).
        t = 0
        state_action_first_t = None if update_upon_every_visit else {}
        t_state_action_reward = []
        total_reward = 0.0
        while not state.terminal and (environment.T is None or t < environment.T):

            a = agent.act(t)
            state_a = (state, a)

            if state_value_plot is not None:
                state_value_plot.update(np.array([v_S[state].get_value()]))

            # mark time step of first visit, if we're doing first-visit evaluation.
            if state_action_first_t is not None and state_a not in state_action_first_t:
                state_action_first_t[state_a] = t

            next_state, next_reward = environment.advance(state, t, a, agent)
            t_state_action_reward.append((t, state_a, next_reward))
            total_reward += next_reward.r
            state = next_state
            t += 1

            agent.sense(state, t)

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of g at each time step t to be properly discounted.
        g = 0
        for i, (t, state_a, reward) in enumerate(reversed(t_state_action_reward)):

            g = agent.gamma * g + reward.r

            # if we're doing every-visit, or if the current time step was the first visit to the state-action, then g is
            # the discounted sample value. use it to update the policy.
            if state_action_first_t is None or state_action_first_t[state_a] == t:

                state, a = state_a

                # if we don't have a baseline, then the target is the return.
                if v_S is None:
                    target = g

                # otherwise, update the baseline state-value estimator and set the target to be the difference between
                # observed return and the baseline. actions that produce an above-baseline return will be reinforced.
                else:
                    v_S[state].update(g)
                    v_S.improve()
                    estimate = v_S[state].get_value()
                    target = g - estimate

                policy.append_update(a, state, alpha, target)

        policy.commit_updates()
        episode_reward_averager.update(total_reward)

        episodes_finished = episode_i + 1

        if num_episodes_per_checkpoint is not None and episodes_finished % num_episodes_per_checkpoint == 0:

            resume_args = {
                'agent': agent,
                'policy': policy,
                'environment': environment,
                'num_episodes': num_episodes,
                'update_upon_every_visit': update_upon_every_visit,
                'alpha': alpha,
                'plot_state_value': plot_state_value,
                'v_S': v_S,
                'num_episodes_per_checkpoint': num_episodes_per_checkpoint,
                'checkpoint_path': checkpoint_path,
                'training_pool_directory': training_pool_directory,
                'training_pool_batch_size': training_pool_batch_size
            }

            checkpoint_path_with_index = insert_index_into_path(checkpoint_path, episodes_finished)
            final_checkpoint_path = checkpoint_path_with_index
            with open(checkpoint_path_with_index, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

        if episodes_finished % episodes_per_print == 0:
            logging.info(f'Finished {episodes_finished} of {num_episodes} episode(s).')

        if has_training_pool_batch_size and episodes_finished % training_pool_batch_size == 0:

            # update training pool
            try:
                with open(training_pool_path, 'wb') as training_pool_file:
                    pickle.dump((agent, policy, v_S, episode_reward_averager), training_pool_file)
            except Exception:
                pass

            # scan training pool for the best available agent
            best_pool_agent = None
            best_pool_policy = None
            best_pool_v_S = None
            best_pool_reward_averager = None
            for training_pool_filename in os.listdir(training_pool_directory):
                try:
                    with open(os.path.join(training_pool_directory, training_pool_filename), 'rb') as f:
                        (
                            pool_agent,
                            pool_policy,
                            pool_v_S,
                            pool_reward_averager
                        ) = pickle.load(f)

                    if best_pool_reward_averager is None or pool_reward_averager.average > best_pool_reward_averager.average:
                        (
                            best_pool_agent,
                            best_pool_policy,
                            best_pool_v_S,
                            best_pool_reward_averager
                        ) = (pool_agent, pool_policy, pool_v_S, pool_reward_averager)
                except Exception:
                    pass

            if best_pool_agent is None:
                logging.info('The training pool contained no agents.')
            else:

                logging.info(f'Best agent in the training pool has an average reward of {best_pool_reward_averager.average:.1f}, compared with the current reward of {episode_reward_averager.average:.1f}.')

                # become the best agent in the pool if it's better than we currently are
                if best_pool_reward_averager.average > episode_reward_averager.average:
                    logging.info('Becoming the better agent.')
                    agent = best_pool_agent
                    policy = best_pool_policy
                    v_S = best_pool_v_S
                    episode_reward_averager = IncrementalSampleAverager()

                    # set the environment reference in continuous-action policies, as we don't pickle it.
                    if isinstance(agent.pi, ContinuousActionPolicy):
                        agent.pi.environment = environment
                else:
                    logging.info('Staying with the current agent.')

    logging.info(f'Completed optimization. Average reward per episode:  {episode_reward_averager.get_value()}')

    return final_checkpoint_path
