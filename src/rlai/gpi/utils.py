import logging
import os
import pickle
import statistics
import threading
import time
from typing import Dict, List, Callable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from rlai.core import MdpAgent
from rlai.policy_gradient.policies.continuous_action import ContinuousActionPolicy

# plotting data and objects
_iteration_ax: Optional[plt.Axes] = None
_iteration_average_reward: Optional[List[float]] = None
_iteration_average_reward_line: Optional[plt.Line2D] = None

_state_space_ax: Optional[plt.Axes] = None
_iteration_total_states: Optional[List[int]] = None
_iteration_total_states_line: Optional[plt.Line2D] = None
_iteration_num_states_improved: Optional[List[int]] = None
_iteration_num_states_improved_line: Optional[plt.Line2D] = None

_time_ax: Optional[plt.Axes] = None
_elapsed_seconds_average_rewards: Optional[Dict[int, List[float]]] = None
_elapsed_seconds_average_reward_line: Optional[plt.Line2D] = None

# the above can be read/written from multiple threads. use lock.
_plot_data_lock = threading.Lock()


def update_policy_iteration_plot():
    """
    Update the policy iteration plot. Can only be done from the main thread.
    """

    if threading.current_thread() != threading.main_thread():
        raise ValueError('Can only update plot on main thread.')

    with _plot_data_lock:

        # plot data will be None prior to the first call to plot_policy_iteration
        if _iteration_average_reward is None:
            return

        iterations = list(range(1, len(_iteration_average_reward) + 1))

        assert _iteration_average_reward_line is not None
        _iteration_average_reward_line.set_data(iterations, _iteration_average_reward)

        assert _iteration_ax is not None
        _iteration_ax.relim()
        _iteration_ax.autoscale_view()

        assert _iteration_total_states_line is not None
        assert _iteration_total_states is not None
        _iteration_total_states_line.set_data(iterations, _iteration_total_states)

        assert _iteration_num_states_improved_line is not None
        assert _iteration_num_states_improved is not None
        _iteration_num_states_improved_line.set_data(iterations, _iteration_num_states_improved)

        assert _state_space_ax is not None
        _state_space_ax.relim()
        _state_space_ax.autoscale_view()

        assert _elapsed_seconds_average_rewards is not None
        seconds, averages = get_second_averages(_elapsed_seconds_average_rewards)

        assert _elapsed_seconds_average_reward_line is not None
        _elapsed_seconds_average_reward_line.set_data(seconds, averages)

        assert _time_ax is not None
        _time_ax.relim()
        _time_ax.autoscale_view()


def plot_policy_iteration(
        iteration_average_reward: List[float],
        iteration_total_states: List[int],
        iteration_num_states_improved: List[int],
        elapsed_seconds_average_rewards: Dict[int, List[float]],
        pdf: Optional[PdfPages]
) -> Optional[plt.Figure]:
    """
    Plot status of policy iteration.

    :param iteration_average_reward: Average reward per iteration.
    :param iteration_total_states: Total number of states per iteration.
    :param iteration_num_states_improved: Number of states improved per iteration.
    :param elapsed_seconds_average_rewards: Elapsed seconds and average rewards.
    :param pdf: PDF for plots.
    :return: Figure if one was created; otherwise, None. The latter will be returned if the caller is not running on the
    main thread. In this case, the plotting data will simply be stored for retrieval by get_policy_iteration_plot_data.
    """

    global _iteration_ax
    global _iteration_average_reward
    global _iteration_average_reward_line

    global _state_space_ax
    global _iteration_total_states
    global _iteration_total_states_line
    global _iteration_num_states_improved
    global _iteration_num_states_improved_line

    global _time_ax
    global _elapsed_seconds_average_rewards
    global _elapsed_seconds_average_reward_line

    # if we're not running on the main thread, then just store the data. someone on the main thread will need to grab
    # it via get_policy_iteration_plot_data for plotting.
    if threading.current_thread() != threading.main_thread():

        # lock and copy data, as caller might soon add to the arrays.
        with _plot_data_lock:
            _iteration_average_reward = iteration_average_reward.copy()
            _iteration_total_states = iteration_total_states.copy()
            _iteration_num_states_improved = iteration_num_states_improved.copy()
            _elapsed_seconds_average_rewards = elapsed_seconds_average_rewards.copy()

        # sleep to let others threads (e.g., the main thread) plot if needed.
        time.sleep(0.01)

        return None

    # noinspection PyTypeChecker
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 6))

    assert isinstance(axs, np.ndarray)

    # reward per iteration
    _iteration_ax = axs[0]
    assert isinstance(_iteration_ax, plt.Axes)
    iterations = list(range(1, len(iteration_average_reward) + 1))
    _iteration_average_reward_line, = _iteration_ax.plot(iterations, iteration_average_reward, '-', label='average')
    _iteration_ax.set_xlabel('Iteration')
    _iteration_ax.set_ylabel('Average total undiscounted reward per episode within iteration')
    _iteration_ax.legend(loc='upper left')
    _iteration_ax.grid()

    # twin-x states per iteration
    _state_space_ax = _iteration_ax.twinx()  # type: ignore[assignment]
    assert _state_space_ax is not None
    _iteration_total_states_line, = _state_space_ax.plot(
        iterations,
        iteration_total_states,
        '--',
        color='orange',
        label='total'
    )
    _iteration_num_states_improved_line, = _state_space_ax.plot(
        iterations,
        iteration_num_states_improved,
        '-',
        color='orange',
        label='improved'
    )
    _state_space_ax.set_yscale('log')
    _state_space_ax.set_ylabel('# states')
    _state_space_ax.legend(loc='center right')

    # reward over elapsed time
    _time_ax = axs[1]
    assert isinstance(_time_ax, plt.Axes)
    seconds, averages = get_second_averages(elapsed_seconds_average_rewards)
    _elapsed_seconds_average_reward_line, = _time_ax.plot(seconds, averages, '-', label='average')
    _time_ax.set_xlabel('Elapsed time (second intervals)')
    _time_ax.set_ylabel('Average total undiscounted reward per episode within interval')
    _time_ax.legend()
    _time_ax.grid()

    plt.tight_layout()

    if pdf is None:
        plt.show(block=False)
    else:
        pdf.savefig()

    plt.close()

    return fig


def get_second_averages(
    elapsed_seconds_average_rewards: Dict[int, List[float]]
) -> Tuple[List[int], List[float]]:
    """
    Get second-averages of rewards.

    :param elapsed_seconds_average_rewards: Elapsed seconds and their average rewards.
    :return: 2-tuple of (1) second values and (2) reward averages.
    """

    seconds = list(sorted(elapsed_seconds_average_rewards.keys()))
    averages = [
        statistics.mean(elapsed_seconds_average_rewards[s])
        for s in seconds
    ]

    return (
        seconds,
        averages
    )


def resume_from_checkpoint(
        checkpoint_path: str,
        resume_function: Callable,
        resume_args_mutator: Optional[Callable[[Dict], None]] = None,
        **new_args
) -> Tuple[Optional[str], MdpAgent]:
    """
    Resume the execution of a previous optimization based on a stored checkpoint.

    :param checkpoint_path: Path to checkpoint file.
    :param resume_function: Function to resume.
    :param resume_args_mutator: A function called prior to resumption. This function will be passed a dictionary of
    arguments comprising the checkpoint. The passed function can change these arguments if desired.
    :param new_args: As a simpler alternative to `resume_args_mutator`, pass any keyword arguments that should replace
    those in the checkpoint. Only those with non-None values will be used.
    :return: 2-tuple of (1) final checkpoint path, or None if checkpoints were not saved, and (2) the updated agent.
    """

    logging.info('Reading checkpoint file to resume.')
    with open(os.path.expanduser(checkpoint_path), 'rb') as checkpoint_file:
        resume_args = pickle.load(checkpoint_file)

    if new_args is not None:
        resume_args.update({
            arg: v
            for arg, v in new_args.items()
            if v is not None
        })

    if resume_args_mutator is not None:
        resume_args_mutator(resume_args)

    # manually set the environment on continuous action policies, as they require a reference but do not pickle it.
    agent = resume_args['agent']
    if hasattr(agent, 'pi') and isinstance(agent.pi, ContinuousActionPolicy):
        agent.pi.environment = resume_args['environment']

    final_checkpoint_path = resume_function(**resume_args)

    # some environments (e.g., gym) hold resources that need to be released
    resume_args['environment'].close()

    return final_checkpoint_path, resume_args['agent']
