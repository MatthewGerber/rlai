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
from rlai.utils import IncrementalSampleAverager, RunThreadManager, ScatterPlot, insert_index_into_path
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
        training_pool_directory: Optional[str] = None
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
    :return: Final checkpoint path, or None if checkpoints were not saved.
    """

    if thread_manager is not None:
        warnings.warn('This optimization method will ignore the thread_manager.')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    if training_pool_directory is not None:
        training_pool_directory = os.path.expanduser(training_pool_directory)
        if not os.path.exists(training_pool_directory):
            os.mkdir(training_pool_directory)

    state_value_plot = None
    if plot_state_value and v_S is not None:
        state_value_plot = ScatterPlot('REINFORCE:  State Value', ['Estimate'], None)

    logging.info(f'Running Monte Carlo-based REINFORCE improvement for {num_episodes} episode(s).')

    episode_reward_averager = IncrementalSampleAverager()
    episodes_per_print = max(1, int(num_episodes * 0.05))
    final_checkpoint_path = None
    processed_training_pool_paths = set()
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

        t_state_action_reward_list = [(t_state_action_reward, state_action_first_t)]

        if training_pool_directory is not None:

            # add current episode to training pool
            pool_path = tempfile.NamedTemporaryFile(delete=False, dir=training_pool_directory).name
            processed_training_pool_paths.add(pool_path)
            with open(pool_path, 'wb') as f:
                pickle.dump((t_state_action_reward, state_action_first_t), f)

            # read available episodes from training pool
            num_training_pool_paths = 0
            for pool_filename in os.listdir(training_pool_directory):
                pool_path = os.path.join(training_pool_directory, pool_filename)
                if pool_path not in processed_training_pool_paths:
                    try:
                        with open(pool_path, 'rb') as f:
                            t_state_action_reward_list.append(pickle.load(f))
                        processed_training_pool_paths.add(pool_path)
                        num_training_pool_paths += 1
                    except Exception as e:
                        logging.error(f'Failed to read training pool path {pool_path}:  {e}')

            logging.info(f'Read {num_training_pool_paths} training pool path(s).')

        for t_state_action_reward, state_action_first_t in t_state_action_reward_list:

            # work backwards through the trace to calculate discounted returns. need to work backward in order for the
            # value of g at each time step t to be properly discounted.
            g = 0
            for i, (t, state_a, reward) in enumerate(reversed(t_state_action_reward)):

                g = agent.gamma * g + reward.r

                # if we're doing every-visit, or if the current time step was the first visit to the state-action, then
                # g is the discounted sample value. use it to update the policy.
                if state_action_first_t is None or state_action_first_t[state_a] == t:

                    state, a = state_a

                    # if we don't have a baseline, then the target is the return.
                    if v_S is None:
                        target = g

                    # otherwise, update the baseline state-value estimator and set the target to be the difference
                    # between observed return and the baseline. actions that produce an above-baseline return will be
                    # reinforced.
                    else:
                        v_S[state].update(g)
                        v_S.improve()
                        estimate = v_S[state].get_value()
                        target = g - estimate

                    policy.append_update(a, state, alpha, target)

        policy.commit_updates()
        episode_reward_averager.update(total_reward)

        if num_episodes_per_checkpoint is not None and episode_i % num_episodes_per_checkpoint == 0:

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
                'training_pool_directory': training_pool_directory
            }

            checkpoint_path_with_index = insert_index_into_path(checkpoint_path, episode_i)
            final_checkpoint_path = checkpoint_path_with_index
            with open(checkpoint_path_with_index, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

        episodes_finished = episode_i + 1
        if episodes_finished % episodes_per_print == 0:
            logging.info(f'Finished {episodes_finished} of {num_episodes} episode(s).')

    logging.info(f'Completed optimization. Average reward per episode:  {episode_reward_averager.get_value()}')

    return final_checkpoint_path
