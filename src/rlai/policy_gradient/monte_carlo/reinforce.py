import logging
import os
import pickle
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from os.path import join
from typing import Optional, Tuple

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
        training_pool_count: Optional[int] = None,
        training_pool_iterate_episodes: Optional[int] = None,
        training_pool_evaluate_episodes: Optional[int] = None
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
    :param training_pool_count: Number of runners in the training pool.
    :param training_pool_iterate_episodes: Number of episodes per training pool iteration.
    :param training_pool_evaluate_episodes: Number of episodes to evaluate the agent when iterating the training pool.
    :return: Final checkpoint path, or None if checkpoints were not saved.
    """

    if thread_manager is not None:
        warnings.warn('This optimization method will ignore the thread_manager.')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    # prepare training pool
    use_training_pool = False
    training_pool_path = None
    training_pool_iteration = None
    has_training_pool_directory = training_pool_directory is not None
    has_training_pool_update_episodes = training_pool_iterate_episodes is not None
    if has_training_pool_directory != has_training_pool_update_episodes:
        raise ValueError('Both training pool directory and episodes must be provided, or neither.')
    elif has_training_pool_directory:

        training_pool_directory = os.path.expanduser(training_pool_directory)

        # create training pool directory. there's a race condition with others in the pool, where another runner
        # might create the directory between the time we check for it here and attempt to make it. calling mkdir when
        # the directory exists will raise an exception.
        try:
            if not os.path.exists(training_pool_directory):
                os.mkdir(training_pool_directory)
        except Exception:
            pass

        use_training_pool = True
        training_pool_path = tempfile.NamedTemporaryFile(dir=training_pool_directory, delete=True).name
        training_pool_iteration = 1

    state_value_plot = None
    if plot_state_value and v_S is not None:
        state_value_plot = ScatterPlot('REINFORCE:  State Value', ['Estimate'], None)

    logging.info(f'Running Monte Carlo-based REINFORCE improvement for {num_episodes} episode(s).')

    start_timestamp = datetime.now()
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
        episodes_finished = episode_i + 1

        if use_training_pool and episodes_finished % training_pool_iterate_episodes == 0:
            policy, v_S = iterate_training_pool(
                training_pool_directory,
                training_pool_path,
                training_pool_count,
                training_pool_iteration,
                training_pool_evaluate_episodes,
                agent,
                policy,
                environment,
                v_S
            )
            agent.pi = policy
            training_pool_iteration += 1

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
                'training_pool_count': training_pool_count,
                'training_pool_iterate_episodes': training_pool_iterate_episodes,
                'training_pool_evaluate_episodes': training_pool_evaluate_episodes
            }

            checkpoint_path_with_index = insert_index_into_path(checkpoint_path, episodes_finished)
            final_checkpoint_path = checkpoint_path_with_index
            with open(checkpoint_path_with_index, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

        elapsed_minutes = (datetime.now() - start_timestamp).total_seconds() / 60.0
        episodes_per_minute = episodes_finished / elapsed_minutes
        estimated_completion_timestamp = start_timestamp + timedelta(minutes=(num_episodes / episodes_per_minute))
        logging.info(f'Finished {episodes_finished} of {num_episodes} episode(s) @ {episodes_per_minute:.1f}/min. Estimated completion:  {estimated_completion_timestamp}.')

    # select final policy from training pool
    if use_training_pool:
        policy, v_S = iterate_training_pool(
            training_pool_directory,
            training_pool_path,
            training_pool_count,
            training_pool_iteration,
            training_pool_evaluate_episodes,
            agent,
            policy,
            environment,
            v_S
        )
        agent.pi = policy

    logging.info('Completed optimization.')

    return final_checkpoint_path


def iterate_training_pool(
        training_pool_directory: str,
        training_pool_path: str,
        training_pool_count: int,
        training_pool_iteration: int,
        training_pool_evaluate_episodes: int,
        agent: StochasticMdpAgent,
        policy: ParameterizedPolicy,
        environment: MdpEnvironment,
        v_S: StateValueEstimator
) -> Tuple[ParameterizedPolicy, StateValueEstimator]:
    """
    Iterate the training pool. This entails evaluating the current agent without updating its policy, waiting for all
    runners in the pool to do the same, and then selecting the best policy from all runners.

    :param training_pool_directory: Training pool directory for all runners.
    :param training_pool_path: Training pool path for the current runner.
    :param training_pool_count: Number of runners in the pool.
    :param training_pool_iteration: Iteration to perform.
    :param training_pool_evaluate_episodes: Number of episodes to evaluate the current agent.
    :param agent: Agent to evaluate.
    :param policy: Current policy.
    :param environment: Environment.
    :param v_S: State-value estimator.
    :return: 2-tuple of the new policy and state-value estimator.
    """

    # evaluate the current agent without updating it
    logging.info('Evaluating agent for training pool.')
    evaluation_start_timestamp = datetime.now()
    evaluation_averager = IncrementalSampleAverager()
    for _ in range(training_pool_evaluate_episodes):
        state = environment.reset_for_new_run(agent)
        agent.reset_for_new_run(state)
        total_reward = 0.0
        t = 0
        while not state.terminal and (environment.T is None or t < environment.T):
            a = agent.act(t)
            next_state, next_reward = environment.advance(state, t, a, agent)
            total_reward += next_reward.r
            state = next_state
            t += 1
            agent.sense(state, t)

        evaluation_averager.update(total_reward)

    logging.info(f'Evaluated agent in {(datetime.now() - evaluation_start_timestamp).total_seconds():.1f} seconds. Average total reward:  {evaluation_averager.average:.2f}')

    # write the policy and its performance to the pool for the current iteration
    with open(f'{training_pool_path}_{training_pool_iteration}', 'wb') as training_pool_file:
        pickle.dump((policy, v_S, evaluation_averager.average), training_pool_file)

    # select policy from current iteration of all runners
    (
        policy,
        v_S
    ) = select_policy_from_training_pool(
        training_pool_directory,
        training_pool_count,
        training_pool_iteration,
        environment
    )

    return policy, v_S


def select_policy_from_training_pool(
        training_pool_directory: str,
        training_pool_count: int,
        training_pool_iteration: int,
        environment: MdpEnvironment
) -> Optional[Tuple[ParameterizedPolicy, StateValueEstimator]]:
    """
    Select a policy from the training pool.

    :param training_pool_directory: Training pool directory.
    :param training_pool_count: Training pool count.
    :param training_pool_iteration: Training pool iteration.
    :param environment: Environment.
    :return: 2-tuple of policy and state-value estimator.
    """

    # wait for all pickles to appear for the current iteration
    training_pool_pickles = []
    while len(training_pool_pickles) != training_pool_count:
        logging.info(f'Waiting for pickles to appear for training pool iteration {training_pool_iteration}.')
        time.sleep(1.0)
        training_pool_pickles.clear()
        for training_pool_filename in filter(lambda s: s.endswith(f'_{training_pool_iteration}'), os.listdir(training_pool_directory)):
            try:
                with open(join(training_pool_directory, training_pool_filename), 'rb') as f:
                    training_pool_pickles.append(pickle.load(f))
            except Exception:
                pass

        logging.info(f'Found {len(training_pool_pickles)} pickle(s).')

    # select best policy
    best_policy = None
    best_v_S = None
    best_average_return = None
    for training_pool_pickle in training_pool_pickles:

        _, _, average_return = training_pool_pickle

        if best_average_return is None or average_return > best_average_return:

            (
                best_policy,
                best_v_S,
                best_average_return
            ) = training_pool_pickle

            # set the environment reference in continuous-action policies, as we don't pickle it.
            if isinstance(best_policy, ContinuousActionPolicy):
                best_policy.environment = environment

    # delete pickles from the previous iteration. can't delete them from the current iteration because other runners
    # might still be scanning them.
    for training_pool_filename in filter(lambda s: s.endswith(f'_{training_pool_iteration - 1}'), os.listdir(training_pool_directory)):
        try:
            os.unlink(join(training_pool_directory, training_pool_filename))
        except Exception:
            pass

    logging.info(f'Selected policy for training pool iteration {training_pool_iteration} with an average return of {best_average_return:.2f}.')

    return best_policy, best_v_S
