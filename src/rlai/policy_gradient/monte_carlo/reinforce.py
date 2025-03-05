import logging
import os
import pickle
import re
import tempfile
import time
import warnings
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import groupby
from os.path import join, expanduser
from typing import Optional, Tuple, List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from rlai.core import MdpState, Action, Reward
from rlai.core.environments.mdp import ContinuousMdpEnvironment
from rlai.docs import rl_text
from rlai.models.sklearn import SKLearnSGD
from rlai.policy_gradient import ParameterizedMdpAgent
from rlai.policy_gradient.policies import ParameterizedPolicy
from rlai.policy_gradient.policies.continuous_action import ContinuousActionPolicy
from rlai.state_value import StateValueEstimator
from rlai.state_value.function_approximation import ApproximateStateValueEstimator
from rlai.utils import (
    IncrementalSampleAverager,
    RunThreadManager,
    insert_index_into_path
)


class Step:
    """
    Bookkeeping values for a step.
    """

    def __init__(
            self,
            t: int,
            state: MdpState,
            action: Action,
            reward: Reward,
            gamma: float
    ):
        """
        Initialize the step.

        :param t: Time step.
        :param state: State.
        :param action: Action.
        :param reward: Reward.
        :param gamma: Gamma (discount).
        """

        self.t = t
        self.state = state
        self.action = action
        self.reward = reward
        self.gamma = gamma

        self.returns: Optional[Returns] = None


class Returns:
    """
    Bookkeeping values for returns at a step.
    """

    def __init__(
            self,
            return_value: float,
            baseline_return_value: float,
            target: float
    ):
        """
        Initialize the return values.

        :param return_value: Return value.
        :param baseline_return_value: Baseline value.
        :param target: Target resulting from return and baseline.
        """

        self.return_value = return_value
        self.baseline_return_value = baseline_return_value
        self.target = target


@rl_text(chapter=13, page=326)
def improve(
        agent: ParameterizedMdpAgent,
        environment: ContinuousMdpEnvironment,
        num_episodes: int,
        update_upon_every_visit: bool,
        alpha: float,
        thread_manager: RunThreadManager,
        plot_state_value: bool,
        num_warmup_episodes: Optional[int] = None,
        num_episodes_per_policy_update_plot: Optional[int] = None,
        policy_update_plot_pdf_directory: Optional[str] = None,
        num_episodes_per_checkpoint: Optional[int] = None,
        checkpoint_path: Optional[str] = None,
        training_pool_directory: Optional[str] = None,
        training_pool_count: Optional[int] = None,
        training_pool_iterate_episodes: Optional[int] = None,
        training_pool_evaluate_episodes: Optional[int] = None,
        training_pool_max_iterations_without_improvement: Optional[int] = None,
        start_episode: Optional[int] = None
) -> Optional[str]:
    """
    Perform Monte Carlo improvement of an agent's policy within an environment via the REINFORCE policy gradient method.
    This improvement function operates over rewards obtained at the end of episodes, so it is only appropriate for
    episodic tasks.

    :param agent: Agent containing a parameterized policy to be optimized.
    :param environment: Environment.
    :param num_episodes: Number of episodes to execute.
    :param update_upon_every_visit: True to update each state-action pair upon each visit within an episode, or False to
    update each state-action pair upon the first visit within an episode.
    :param alpha: Policy gradient step size.
    :param thread_manager: Thread manager. The current function (and the thread running it) will wait on this manager
    before starting each iteration. This provides a mechanism for pausing, resuming, and aborting training. Omit for no
    waiting.
    :param plot_state_value: Whether to plot the state-value.
    :param num_warmup_episodes: Number of warmup episodes to run before updating the policy. Warmup episodes allow
    estimates (e.g., means and variances of feature scalers, baseline state-value estimators, etc.) to settle before
    updating the policy.
    :param num_episodes_per_policy_update_plot: Number of episodes per plot.
    :param policy_update_plot_pdf_directory: Directory in which to store plot PDFs, or None to display them directly.
    :param num_episodes_per_checkpoint: Number of episodes per checkpoint save.
    :param checkpoint_path: Checkpoint path. Must be provided if `num_episodes_per_checkpoint` is provided.
    :param training_pool_directory: Path to directory in which to store pooled training runs.
    :param training_pool_count: Number of runners in the training pool.
    :param training_pool_iterate_episodes: Number of episodes per training pool iteration.
    :param training_pool_evaluate_episodes: Number of episodes to evaluate the agent when iterating the training pool.
    :param training_pool_max_iterations_without_improvement: Maximum number of training pool iterations to allow
    before reverting to the best prior agent, or None to never revert.
    :param start_episode: 1-based episode to start at, or None to start at episode 1.
    :return: Final checkpoint path, or None if checkpoints were not saved.
    """

    if thread_manager is not None:
        warnings.warn('This optimization method will ignore the thread_manager.')

    if checkpoint_path is not None:
        checkpoint_path = os.path.expanduser(checkpoint_path)

    if policy_update_plot_pdf_directory is not None:
        policy_update_plot_pdf_directory = os.path.expanduser(policy_update_plot_pdf_directory)

    # we work backward through the episode when updating the baseline model. if we have a baseline model, then indicate
    # this so that plotting is ordered correctly.
    if isinstance(agent.v_S, ApproximateStateValueEstimator) and isinstance(agent.v_S.model, SKLearnSGD):
        agent.v_S.model.reverse_time_steps = True

    assert isinstance(agent.pi, ParameterizedPolicy)

    training_pool = TrainingPool.init(
        agent=agent,
        environment=environment,
        training_pool_directory=training_pool_directory,
        training_pool_count=training_pool_count,
        training_pool_iterate_episodes=training_pool_iterate_episodes,
        training_pool_evaluate_episodes=training_pool_evaluate_episodes,
        training_pool_max_iterations_without_improvement=training_pool_max_iterations_without_improvement
    )

    state_value_plot = None
    if plot_state_value and agent.v_S is not None:

        # local-import so that we don't crash on raspberry pi os, where we can't install qt6.
        from rlai.plot_utils import ScatterPlot

        state_value_plot = ScatterPlot('REINFORCE:  State Value', ['Estimate'], None)

    logging.info(f'Running Monte Carlo-based REINFORCE improvement for {num_episodes} episode(s).')

    start_timestamp = datetime.now()
    final_checkpoint_path = None

    if start_episode is None:
        episodes_finished = 0
    else:
        episodes_finished = start_episode - 1
        logging.info(f'Starting with episode {start_episode}.')

    while episodes_finished < num_episodes:

        # reset the environment for the new run (always use the agent we're learning about, as state identifiers come
        # from it), and reset the agent accordingly.
        state = environment.reset_for_new_run(agent)
        agent.reset_for_new_run(state)

        # simulate until episode termination, keeping a trace of state-action pairs and their immediate rewards, as well
        # as the times of their first visits (only if we're doing first-visit evaluation).
        t = 0
        state_action_first_t: Optional[Dict[Tuple[MdpState, Action], int]] = None if update_upon_every_visit else {}
        steps = []
        truncation_time_step = None
        while not state.terminal:
            try:

                if state.truncated and truncation_time_step is None:
                    truncation_time_step = t
                    logging.info(f'Episode was truncated after {t} step(s).')
                elif not state.truncated and truncation_time_step is not None:
                    raise ValueError('Truncation cannot be exited.')

                a = agent.act(t)
                state_a = (state, a)

                if state_value_plot is not None and agent.v_S is not None:
                    state_value_plot.update(np.array([agent.v_S[state].get_value()]))

                # mark time step of first visit, if we're doing first-visit evaluation.
                if state_action_first_t is not None and state_a not in state_action_first_t:
                    state_action_first_t[state_a] = t

                next_state, next_reward = environment.advance(state, t, a, agent)
                gamma = agent.gamma
                steps.append(Step(t, state, a, next_reward, gamma))
                state = next_state
                t += 1
                agent.sense(state, t)

                # if we've truncated and the discount has converged to zero, then the return at the truncation time will
                # not change by running longer. we've got an accurate return estimate at truncation. exit the episode.
                if truncation_time_step is not None:
                    num_post_truncation_steps = (t - truncation_time_step)
                    post_truncation_discount = gamma ** num_post_truncation_steps
                    if np.isclose(post_truncation_discount, 0.0):
                        raise ValueError(
                            f'Post-truncation discount converged to zero after {num_post_truncation_steps} step(s).'
                        )

            # if anything blows up, then let the environment know that we are exiting the episode.
            except ValueError as e:
                logging.info(f'{e} Exiting episode without termination.')
                environment.exiting_episode_without_termination()
                break

        # add metrics to per-episode collection for easy plotting
        for metric, value in environment.metric_value.items():
            if metric not in environment.metric_episode_value:
                environment.metric_episode_value[metric] = {}
            environment.metric_episode_value[metric][episodes_finished] = value

        # work backwards through the trace to calculate discounted returns. need to work backward in order for the value
        # of g at each time step t to be properly discounted.
        g = 0.0
        for step in reversed(steps):

            g = step.gamma * g + step.reward.r

            # only update value estimates before the truncation time step if we have one
            if truncation_time_step is not None and step.t >= truncation_time_step:
                continue

            # if we're doing every-visit, or if the current time step was the first visit to the state-action, then g is
            # the discounted sample value. use it to update the policy.
            if state_action_first_t is None or (
                state_action_first_t[(step.state, step.action)] == step.t
            ):
                # if we don't have a baseline, then the baseline return is zero and the target is the return.
                if agent.v_S is None:
                    baseline_return = 0.0

                # otherwise, the baseline return is the current estimate of the return from the state. update the
                # state-value estimator with the obtained return. this only adds the example to the estimator. it does
                # not improve the estimator.
                else:
                    baseline_return = agent.v_S[step.state].get_value()
                    agent.v_S[step.state].update(g)

                # form the target as difference between observed return and baseline. actions that produce an
                # above-baseline return will be reinforced.
                target = g - baseline_return

                # append the update to the policy. this does not commit the updates.
                if num_warmup_episodes is None or episodes_finished > num_warmup_episodes:
                    agent.pi.append_update(step.action, step.state, alpha, target)

                step.returns = Returns(
                    return_value=g,
                    baseline_return_value=baseline_return,
                    target=target
                )

        # improve the state-value estimator with the updates that were provided
        if agent.v_S is not None:
            agent.v_S.improve()

        # commit the updates to the policy
        if num_warmup_episodes is None or episodes_finished > num_warmup_episodes:
            agent.pi.commit_updates()

        episodes_finished += 1

        # plot policy update
        if (
            num_episodes_per_policy_update_plot is not None and
            episodes_finished % num_episodes_per_policy_update_plot == 0
        ):
            # initialize pdf if we're saving to a directory
            if policy_update_plot_pdf_directory is None:
                pdf = None
            else:
                os.makedirs(policy_update_plot_pdf_directory, exist_ok=True)
                pdf = PdfPages(os.path.expanduser(os.path.join(
                    policy_update_plot_pdf_directory,
                    f'reinforce_{episodes_finished}.pdf'
                )))

            if agent.v_S is not None:
                agent.v_S.plot(pdf)

            num_steps_per_plot_group = 500
            for _, plot_step_grouper in groupby(steps, key=lambda s: s.t // num_steps_per_plot_group):

                group_steps = list(plot_step_grouper)
                group_t = [step.t for step in group_steps]

                group_start_t = group_steps[0].t
                group_end_t = group_steps[-1].t

                figure_size = (0.05 * len(group_steps), 0.025 * len(group_steps))
                plt.figure(figsize=figure_size)

                group_non_truncated_steps = [step for step in group_steps if step.returns is not None]
                group_non_truncated_steps_t = [step.t for step in group_non_truncated_steps]

                # plot rewards and returns
                plt.plot(
                    group_t,
                    [step.reward.r for step in group_steps],
                    color='red',
                    label='Reward:  r(t)',
                    linewidth=0.5
                )
                plt.plot(
                    group_non_truncated_steps_t,
                    [step.returns.return_value for step in group_non_truncated_steps],  # type: ignore[union-attr]
                    color='green',
                    label='Return:  g(t)',
                    linewidth=0.5
                )
                plt.plot(
                    group_non_truncated_steps_t,
                    [step.returns.baseline_return_value for step in group_non_truncated_steps],  # type: ignore[union-attr]
                    color='violet',
                    label='Value:  v(t)',
                    linewidth=0.5
                )
                plt.plot(
                    group_non_truncated_steps_t,
                    [step.returns.target for step in group_non_truncated_steps],  # type: ignore[union-attr]
                    color='orange',
                    label='Target:  g(t) - v(t)',
                    linewidth=0.5
                )
                plt.ylabel('Returns and Rewards')

                # plot any vertical lines in the current group of steps
                for time_step, kwargs in environment.time_step_axv_lines.items():
                    if group_start_t <= time_step <= group_end_t:
                        plt.axvline(time_step, **kwargs)

                plt.xlabel('Time step')
                plt.title(f'Episode {episodes_finished} Steps {group_start_t}-{group_end_t}')
                plt.legend(loc='upper left')

                # plot gamma (discount) in a twin-x axes, as the scale is much different.
                gamma_axe: plt.Axes = plt.twinx()  # type: ignore[assignment]
                gamma_axe.plot(
                    group_t,
                    [step.gamma for step in group_steps],
                    color='blue',
                    label='gamma(t)',
                    linewidth=0.5
                )
                gamma_axe.set_ylabel('Gamma')
                gamma_axe.legend(loc='upper right')

                plt.tight_layout()

                if pdf is None:
                    plt.show()
                else:
                    pdf.savefig()

                plt.close()

                # plot any data added by the environment for the current group of steps
                if len(environment.plot_label_data_kwargs) > 0:
                    plt.figure(figsize=figure_size)
                    for plot_label, (plot_data, plot_kwargs) in environment.plot_label_data_kwargs.items():
                        plot_data_steps = [t for t in plot_data if group_start_t <= t <= group_end_t]
                        plt.plot(
                            plot_data_steps,
                            [plot_data[t] for t in plot_data_steps],
                            label=plot_label,
                            **plot_kwargs
                        )
                    plt.grid()
                    plt.legend()
                    plt.tight_layout()

                    if pdf is None:
                        plt.show()
                    else:
                        pdf.savefig()

                    plt.close()

            # plot per-episode metrics
            if len(environment.metric_episode_value) > 0:
                for metric in environment.metric_episode_value:
                    plt.plot(
                        list(environment.metric_episode_value[metric].keys()),
                        list(environment.metric_episode_value[metric].values()),
                        label=metric
                    )
                    plt.xlabel('Episode')
                    plt.ylabel('Value')
                    plt.grid()
                    plt.legend()
                    plt.tight_layout()

                if pdf is None:
                    plt.show()
                else:
                    pdf.savefig()

                plt.close()

            if pdf is not None:
                pdf.close()

        num_fallback_iterations = 0
        if (
            training_pool is not None and
            training_pool_iterate_episodes is not None and
            episodes_finished % training_pool_iterate_episodes == 0
        ):
            num_fallback_iterations = training_pool.iterate(False)

        if num_episodes_per_checkpoint is not None and episodes_finished % num_episodes_per_checkpoint == 0:

            assert checkpoint_path is not None

            resume_args = {
                'agent': agent,
                'environment': environment,
                'num_episodes': num_episodes,
                'update_upon_every_visit': update_upon_every_visit,
                'alpha': alpha,
                'plot_state_value': plot_state_value,
                'num_episodes_per_policy_update_plot': num_episodes_per_policy_update_plot,
                'policy_update_plot_pdf_directory': policy_update_plot_pdf_directory,
                'num_episodes_per_checkpoint': num_episodes_per_checkpoint,
                'checkpoint_path': checkpoint_path,
                'training_pool_directory': training_pool_directory,
                'training_pool_count': training_pool_count,
                'training_pool_iterate_episodes': training_pool_iterate_episodes,
                'training_pool_evaluate_episodes': training_pool_evaluate_episodes,
                'training_pool_max_iterations_without_improvement': training_pool_max_iterations_without_improvement,
                'start_episode': episodes_finished + 1
            }

            checkpoint_path_with_index = insert_index_into_path(checkpoint_path, episodes_finished)
            final_checkpoint_path = checkpoint_path_with_index
            os.makedirs(os.path.dirname(final_checkpoint_path), exist_ok=True)
            with open(checkpoint_path_with_index, 'wb') as checkpoint_file:
                pickle.dump(resume_args, checkpoint_file)

        elapsed_minutes = (datetime.now() - start_timestamp).total_seconds() / 60.0
        episodes_per_minute = episodes_finished / elapsed_minutes
        estimated_completion_timestamp = start_timestamp + timedelta(minutes=(num_episodes / episodes_per_minute))
        logging.info(
            f'Finished {episodes_finished} of {num_episodes} episode(s) @ {episodes_per_minute:.1f}/min. Estimated '
            f'completion:  {estimated_completion_timestamp}.'
        )

        # decrement episodes finished if we fell back to an earlier iteration
        if num_fallback_iterations > 0:
            assert training_pool_iterate_episodes is not None
            num_fallback_episodes = num_fallback_iterations * training_pool_iterate_episodes
            episodes_finished -= num_fallback_episodes
            logging.info(f'Removed {num_fallback_episodes} finished episodes due to a fallback.')

    if training_pool is not None:
        logging.info('Iterating the training pool one final time to ensure that the best policy/v_S is set.')
        training_pool.iterate(True)

    logging.info('Completed optimization.')

    return final_checkpoint_path


class TrainingPool:
    """
    Training pool.
    """

    @staticmethod
    def init(
            agent: ParameterizedMdpAgent,
            environment: ContinuousMdpEnvironment,
            training_pool_directory: Optional[str] = None,
            training_pool_count: Optional[int] = None,
            training_pool_iterate_episodes: Optional[int] = None,
            training_pool_evaluate_episodes: Optional[int] = None,
            training_pool_max_iterations_without_improvement: Optional[int] = None
    ) -> Optional['TrainingPool']:
        """
        Initialize the training pool.

        :param agent: Agent.
        :param environment: Environment.
        :param training_pool_directory: Path to directory in which to store pooled training runs.
        :param training_pool_count: Number of runners in the training pool.
        :param training_pool_iterate_episodes: Number of episodes per training pool iteration.
        :param training_pool_evaluate_episodes: Number of episodes to evaluate the agent when iterating the training
        pool.
        :param training_pool_max_iterations_without_improvement: Maximum number of training pool iterations to allow
        before reverting to the best prior agent.
        :return: Training pool, or None if not configured.
        """

        if (
            training_pool_directory is None and
            training_pool_count is None and
            training_pool_iterate_episodes is None and
            training_pool_evaluate_episodes is None
        ):
            training_pool = None

        elif (
            training_pool_directory is not None and
            training_pool_count is not None and
            training_pool_iterate_episodes is not None and
            training_pool_evaluate_episodes is not None
        ):

            training_pool_directory = os.path.expanduser(training_pool_directory)

            # create training pool directory. there's a race condition with others in the pool, where another runner
            # might create the directory between the time we check for it here and attempt to make it. calling mkdir
            # when the directory exists will raise an exception. pass the exception and proceed, as the directory will
            # exist.
            # noinspection PyBroadException
            try:
                if not os.path.exists(training_pool_directory):
                    os.mkdir(training_pool_directory)
            except Exception:
                pass

            training_pool = TrainingPool(
                agent=agent,
                environment=environment,
                training_pool_directory=training_pool_directory,
                training_pool_count=training_pool_count,
                training_pool_evaluate_episodes=training_pool_evaluate_episodes,
                training_pool_max_iterations_without_improvement=training_pool_max_iterations_without_improvement
            )
        else:
            raise ValueError('Training pool params must all be None or not None.')

        return training_pool

    @staticmethod
    def plot_iterations(
            log_path: str
    ):
        """
        Plot training pool performance by iteration.

        :param log_path: Path to log file.
        """

        iteration_return = {}
        fallback_at_iteration_at_reward_to_iteration_to_reward = {}
        fallback_re = re.compile(
            'INFO:root:At training pool iteration (\\d+): {2}Falling back to policy/v_S at iteration (\\d+) with '
            'average return of ([\\-\\d.]+), after an average return of ([\\-\\d.]+) and .+'
        )
        with open(expanduser(log_path), 'r') as f:
            for line in f:

                # INFO:root:Selected policy for training pool iteration 1 with an average return of 11.84.
                if line.startswith('INFO:root:Selected policy'):
                    s = line.split(' iteration ')[1]
                    iteration = int(s[0:s.index(' ')])
                    avg_return = float(s.split('return of ')[1].strip('.\n'))
                    iteration_return[iteration] = avg_return

                fallback_match = fallback_re.match(line)
                if fallback_match is not None:
                    at_iteration = int(fallback_match.group(1))
                    at_reward = float(fallback_match.group(4))
                    to_iteration = int(fallback_match.group(2))
                    to_reward = float(fallback_match.group(3))
                    fallback_at_iteration_at_reward_to_iteration_to_reward[at_iteration] = (
                        at_reward,
                        to_iteration,
                        to_reward
                    )

        plt.plot(list(iteration_return.keys()), list(iteration_return.values()), label='Pooled REINFORCE')
        to_iteration_color: Dict[int, Any] = {}
        for at_iteration, (at_reward, to_iteration, to_reward) in fallback_at_iteration_at_reward_to_iteration_to_reward.items():
            color = to_iteration_color.get(to_iteration, None)
            lines = plt.plot([at_iteration, to_iteration], [at_reward, to_reward], color=color)
            if color is None:
                to_iteration_color[to_iteration] = lines[0].get_color()

        plt.xlabel('Pool iteration')
        plt.ylabel('Avg. evaluation return')
        plt.title('Pooled learning performance')
        plt.grid()
        plt.legend()
        plt.show()

    def __init__(
            self,
            agent: ParameterizedMdpAgent,
            environment: ContinuousMdpEnvironment,
            training_pool_directory: str,
            training_pool_count: int,
            training_pool_evaluate_episodes: int,
            training_pool_max_iterations_without_improvement: Optional[int]
    ):
        """
        Initialize the training pool.

        :param agent: Agent.
        :param environment: Environment.
        :param training_pool_directory: Path to directory in which to store pooled training runs.
        :param training_pool_count: Number of runners in the training pool.
        :param training_pool_evaluate_episodes: Number of episodes to evaluate the agent when iterating the training
        pool.
        :param training_pool_max_iterations_without_improvement: Maximum number of training pool iterations to allow
        before reverting to the best prior agent, or None to never revert.
        """

        self.agent = agent
        self.environment = environment
        self.training_pool_directory = training_pool_directory
        self.training_pool_count = training_pool_count
        self.training_pool_evaluate_episodes = training_pool_evaluate_episodes
        self.training_pool_max_iterations_without_improvement = training_pool_max_iterations_without_improvement

        self.training_pool_path = tempfile.NamedTemporaryFile(dir=training_pool_directory, delete=True).name
        self.training_pool_iteration = 1
        self.training_pool_iterations_without_improvement = 0
        self.training_pool_best_overall_policy: Optional[ParameterizedPolicy] = None
        self.training_pool_best_overall_v_S: Optional[StateValueEstimator] = None
        self.training_pool_best_overall_average_return: Optional[float] = None
        self.training_pool_best_overall_iteration: Optional[int] = None

    def iterate(
            self,
            greedy: bool
    ) -> int:
        """
        Iterate the training pool. This entails evaluating the current agent without updating its policy, waiting for
        all runners in the pool to do the same, and then updating the agent with the best policy from all runners.

        :param greedy: Whether to set the policy to the best one, either from the current iteration or a past iteration.
        :return: Number of preceding iterations that were erased due to falling back to an earlier iteration.
        """

        # TODO:  Adaptive number of evaluation episodes based on return statistics
        # evaluate the current agent without updating it
        logging.info(f'Evaluating agent for training pool iteration {self.training_pool_iteration}.')
        evaluation_start_timestamp = datetime.now()
        evaluation_averager = IncrementalSampleAverager()
        for _ in range(self.training_pool_evaluate_episodes):
            state = self.environment.reset_for_new_run(self.agent)
            self.agent.reset_for_new_run(state)
            total_reward = 0.0
            t = 0
            truncation_time_step = None
            while not state.terminal:
                try:
                    if state.truncated and truncation_time_step is None:
                        truncation_time_step = t

                    a = self.agent.act(t)
                    next_state, next_reward = self.environment.advance(state, t, a, self.agent)
                    total_reward += next_reward.r
                    state = next_state
                    t += 1
                    self.agent.sense(state, t)

                    # if we've truncated and the discount has converged to zero, then the return at the truncation time will
                    # not change by running longer. we've got an accurate return estimate at truncation. exit the episode.
                    if truncation_time_step is not None:
                        num_post_truncation_steps = (t - truncation_time_step)
                        post_truncation_discount = self.agent.gamma ** num_post_truncation_steps
                        if np.isclose(post_truncation_discount, 0.0):
                            raise ValueError(
                                f'Post-truncation discount converged to zero after {num_post_truncation_steps} step(s).'
                            )

                # if anything blows up, then let the environment know that we are exiting the episode.
                except ValueError as e:
                    logging.info(f'{e} Exiting episode without termination.')
                    self.environment.exiting_episode_without_termination()
                    break

            evaluation_averager.update(total_reward)

        logging.info(
            f'Evaluated agent in {(datetime.now() - evaluation_start_timestamp).total_seconds():.1f} seconds. Average '
            f'total reward:  {evaluation_averager.average:.2f}'
        )

        # write the policy and its performance to the pool for the current iteration
        with open(f'{self.training_pool_path}_{self.training_pool_iteration}', 'wb') as training_pool_file:
            pickle.dump((self.agent.pi, self.agent.v_S, evaluation_averager.average), training_pool_file)

        # select policy from current iteration of all runners
        (
            best_policy,
            best_state_value_estimator,
            best_average_return
        ) = self.select_best()

        # track the policy/v_S with the best average return
        if (
            self.training_pool_best_overall_average_return is None or
            best_average_return > self.training_pool_best_overall_average_return
        ):
            self.training_pool_best_overall_policy = best_policy
            self.training_pool_best_overall_v_S = best_state_value_estimator
            self.training_pool_best_overall_average_return = best_average_return
            self.training_pool_best_overall_iteration = self.training_pool_iteration
            self.training_pool_iterations_without_improvement = 0
            logging.info(
                f'Bookmarked new best policy/v_S at training pool iteration {self.training_pool_iteration} with '
                f'average return of {self.training_pool_best_overall_average_return}.'
            )
        else:
            self.training_pool_iterations_without_improvement += 1
            logging.info(
                f'Training pool iterations without improvement:  {self.training_pool_iterations_without_improvement}'
            )

        # fall back to the best prior policy if we've failed to improve upon it for too many iterations. copy to prevent
        # changes to the original, which might get used again.
        num_fallback_iterations = 0
        if (
            self.training_pool_max_iterations_without_improvement is not None and
            self.training_pool_iterations_without_improvement > self.training_pool_max_iterations_without_improvement
        ):
            logging.info(
                f'At training pool iteration {self.training_pool_iteration}:  Falling back to policy/v_S at iteration '
                f'{self.training_pool_best_overall_iteration} with average return of '
                f'{self.training_pool_best_overall_average_return}, after an average return of {best_average_return} '
                f'and {self.training_pool_iterations_without_improvement} iteration(s) without improvement.'
            )
            self.agent.pi = deepcopy(self.training_pool_best_overall_policy)  # type: ignore[assignment]
            self.agent.v_S = deepcopy(self.training_pool_best_overall_v_S)
            num_fallback_iterations = self.training_pool_iterations_without_improvement
            self.training_pool_iterations_without_improvement = 0

        # greedy iteration always sets to the best overall. copy to prevent changes to the original, which might get
        # used again.
        elif greedy:
            logging.info(
                f'Setting to greedy policy/v_S from iteration {self.training_pool_best_overall_iteration} with average '
                f'return of {self.training_pool_best_overall_average_return}.'
            )
            self.agent.pi = deepcopy(self.training_pool_best_overall_policy)  # type: ignore[assignment]
            self.agent.v_S = deepcopy(self.training_pool_best_overall_v_S)

        # set the agent's policy/v_S to the best available from the pool's current iteration. no need to copy here, as
        # the objects come directly from a pickle.load call.
        else:
            self.agent.pi = best_policy
            self.agent.v_S = best_state_value_estimator

        # set the environment reference in continuous-action policies, as we don't pickle it or deepcopy it.
        if isinstance(self.agent.pi, ContinuousActionPolicy):
            self.agent.pi.environment = self.environment

        self.training_pool_iteration += 1

        return num_fallback_iterations

    def select_best(
            self,
    ) -> Tuple[ParameterizedPolicy, StateValueEstimator, float]:
        """
        Select the best policy from the training pool.

        :return: 3-tuple of the best policy, its state-value estimator, and its average return.
        """

        # wait for all pickles to appear for the current iteration
        training_pool_policy_state_value_returns: List[Tuple] = []
        while len(training_pool_policy_state_value_returns) != self.training_pool_count:
            logging.info(f'Waiting for pickles to appear for training pool iteration {self.training_pool_iteration}.')
            time.sleep(1.0)
            training_pool_policy_state_value_returns.clear()
            for training_pool_filename in filter(
                lambda s: s.endswith(f'_{self.training_pool_iteration}'),
                os.listdir(self.training_pool_directory)
            ):
                # noinspection PyBroadException
                try:
                    with open(join(self.training_pool_directory, training_pool_filename), 'rb') as f:
                        training_pool_policy_state_value_returns.append(pickle.load(f))
                except Exception:
                    pass

            logging.info(f'Read {len(training_pool_policy_state_value_returns)} pickle(s).')

        # select best policy
        best_policy: Optional[ParameterizedPolicy] = None
        best_state_value_estimator: Optional[StateValueEstimator] = None
        best_average_return: Optional[float] = None
        for policy, state_value_estimator, average_return in training_pool_policy_state_value_returns:

            if best_average_return is None or average_return > best_average_return:

                best_policy = policy
                best_state_value_estimator = state_value_estimator
                best_average_return = average_return

        # delete pickles from the previous iteration. we can't delete them from the current iteration because other
        # runners might still be scanning them.
        for training_pool_filename in filter(
            lambda s: s.endswith(f'_{self.training_pool_iteration - 1}'),
            os.listdir(self.training_pool_directory)
        ):
            # noinspection PyBroadException
            try:
                os.unlink(join(self.training_pool_directory, training_pool_filename))
            except Exception:
                pass

        logging.info(
            f'Selected policy for training pool iteration {self.training_pool_iteration} with an average return of '
            f'{best_average_return:.2f}.'
        )

        assert best_policy is not None
        assert best_state_value_estimator is not None
        assert best_average_return is not None

        return best_policy, best_state_value_estimator, best_average_return
