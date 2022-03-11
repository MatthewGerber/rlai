import logging
import os
import pickle
import tempfile
import warnings
from datetime import datetime
from random import shuffle
from typing import Optional, List, Tuple

import numpy as np
from numpy.random import RandomState

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
        training_pool_subpool_size: Optional[int] = None,
        training_pool_update_episodes: Optional[int] = None,
        training_pool_epsilon: Optional[float] = None,
        return_averager_alpha: Optional[float] = None
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
    :param training_pool_subpool_size: Size of the current optimizer's subpool.
    :param training_pool_update_episodes: Number of episodes per training pool update.
    :param training_pool_epsilon: Probability of selecting a random (rather than greedy) element from the training pool
    when updating.
    :param return_averager_alpha: Step size to use in return averager, or None for standard average.
    :return: Final checkpoint path, or None if checkpoints were not saved.
    """

    if thread_manager is not None:
        warnings.warn('This optimization method will ignore the thread_manager.')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    random_state = agent.random_state

    # prepare training pool
    training_subpool_paths = None
    has_training_pool_directory = training_pool_directory is not None
    has_training_pool_update_episodes = training_pool_update_episodes is not None
    if has_training_pool_directory != has_training_pool_update_episodes:
        raise ValueError('Both training pool directory and episodes must be provided, or neither.')
    elif has_training_pool_directory:

        training_pool_directory = os.path.expanduser(training_pool_directory)

        # create training pool directory. there's a race condition with others in the pool, where another might create
        # the directory between the time we check for it here and attempt to make it. calling mkdir when the directory
        # exists will raise an exception.
        try:
            if not os.path.exists(training_pool_directory):
                os.mkdir(training_pool_directory)
        except Exception:
            pass

        training_subpool_paths = [
            tempfile.NamedTemporaryFile(dir=training_pool_directory, delete=False).name
            for _ in range(training_pool_subpool_size)
        ]

    state_value_plot = None
    if plot_state_value and v_S is not None:
        state_value_plot = ScatterPlot('REINFORCE:  State Value', ['Estimate'], None)

    logging.info(f'Running Monte Carlo-based REINFORCE improvement for {num_episodes} episode(s).')

    episode_return_averager = IncrementalSampleAverager(alpha=return_averager_alpha)
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
            state = next_state
            t += 1

            agent.sense(state, t)

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of g at each time step t to be properly discounted.
        g = 0.0
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
        episode_return_averager.update(g)

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
                'training_pool_subpool_size': training_pool_subpool_size,
                'training_pool_update_episodes': training_pool_update_episodes,
                'training_pool_epsilon': training_pool_epsilon,
                'return_averager_alpha': return_averager_alpha
            }

            checkpoint_path_with_index = insert_index_into_path(checkpoint_path, episodes_finished)
            final_checkpoint_path = checkpoint_path_with_index
            with open(checkpoint_path_with_index, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

        if episodes_finished % episodes_per_print == 0:
            logging.info(f'Finished {episodes_finished} of {num_episodes} episode(s).')

        if has_training_pool_update_episodes and episodes_finished % training_pool_update_episodes == 0:

            # replace the worst agent in the subpool with the current agent. but only do this if the current agent's
            # return is better. we don't want to make the subpool worse.
            try:
                worst_subpool_path, worst_subpool_return = get_worst_training_subpool_path(training_subpool_paths)
                if worst_subpool_return is None or episode_return_averager.average > worst_subpool_return:
                    with open(worst_subpool_path, 'wb') as worst_subpool_file:
                        pickle.dump((agent, policy, v_S, episode_return_averager), worst_subpool_file)
            except Exception:
                pass

            # select a new agent from the pool either greedily or randomly
            (
                agent,
                policy,
                v_S,
                episode_return_averager
            ) = select_agent_from_training_pool(training_pool_directory, training_pool_epsilon, random_state, environment)

    logging.info(f'Completed optimization. Average return per episode:  {episode_return_averager.average}')

    return final_checkpoint_path


def get_worst_training_subpool_path(
        training_subpool_paths: List[str]
) -> Tuple[str, Optional[float]]:
    """
    Get the worst training subpool path.

    :param training_subpool_paths: Paths.
    :return: 2-tuple of the worst path and its average return (or None if the worst path did not contain an agent).
    """

    worst_training_subpool_path = None
    worst_average_return = None
    for training_subpool_path in training_subpool_paths:
        try:

            with open(training_subpool_path, 'rb') as f:
                (
                    _,
                    _,
                    _,
                    return_averager
                ) = pickle.load(f)

            if worst_average_return is None or return_averager.average < worst_average_return:
                worst_training_subpool_path = training_subpool_path
                worst_average_return = return_averager.average

        # exception will be thrown if the file hasn't yet been written with an agent. always use such a file.
        except Exception:
            worst_training_subpool_path = training_subpool_path
            break

    return worst_training_subpool_path, worst_average_return


def select_agent_from_training_pool(
        training_pool_directory: str,
        training_pool_epsilon: Optional[float],
        random_state: RandomState,
        environment: MdpEnvironment
) -> Optional[Tuple[StochasticMdpAgent, ParameterizedPolicy, StateValueEstimator, IncrementalSampleAverager]]:
    """
    Select an agent from the training pool.

    :param training_pool_directory: Training pool directory.
    :param training_pool_epsilon: Probability of selecting a random agent from the pool, or None to always select the
    best agent.
    :param random_state: Random state.
    :param environment: Environment.
    :return: 4-tuple of agent, policy, state-value estimator, and return averager.
    """

    select_greedily = training_pool_epsilon is None or random_state.random() >= training_pool_epsilon

    # get agents in random order. we'll take the first if selecting randomly.
    training_pool_filenames = os.listdir(training_pool_directory)
    shuffle(training_pool_filenames, random_state.random)

    # select an entry from the pool either greedily (based on average return) or randomly (for exploration).
    scan_start_datetime = datetime.now()
    selected_pool_agent = None
    selected_pool_policy = None
    selected_pool_v_S = None
    selected_pool_return_averager = None
    for training_pool_filename in training_pool_filenames:
        try:

            with open(os.path.join(training_pool_directory, training_pool_filename), 'rb') as f:
                (
                    pool_agent,
                    pool_policy,
                    pool_v_S,
                    pool_return_averager
                ) = pickle.load(f)

            if selected_pool_return_averager is None or pool_return_averager.average > selected_pool_return_averager.average:

                (
                    selected_pool_agent,
                    selected_pool_policy,
                    selected_pool_v_S,
                    selected_pool_return_averager
                ) = (pool_agent, pool_policy, pool_v_S, pool_return_averager)

                # set the environment reference in continuous-action policies, as we don't pickle it.
                if isinstance(selected_pool_agent.pi, ContinuousActionPolicy):
                    selected_pool_agent.pi.environment = environment

            # take the first (random) entry if we're not selecting greedily
            if not select_greedily:
                break

        # might get exception if another process is writing the current agent
        except Exception:
            pass

    logging.info(f'Scanned training pool in {(datetime.now() - scan_start_datetime).total_seconds():.1f} seconds.')

    if selected_pool_agent is None:
        logging.info('The training pool contained no agents.')
    else:
        logging.info(f'Selected agent {"greedily" if select_greedily else "randomly"} from the training pool with an average return of {selected_pool_return_averager.average:.1f}.')

    return selected_pool_agent, selected_pool_policy, selected_pool_v_S, selected_pool_return_averager
