import math
import os
import pickle
import sys
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.meta import rl_text
from rlai.runners.monitor import Monitor
from rlai.utils import load_class


@rl_text(chapter='Training and Running Agents', page=1)
def run(
        args: List[str]
):
    """
    Run an agent within an environment.

    :param args: Arguments.
    """

    parser = ArgumentParser(
        description='Run an agent within an environment. This does not support learning GPI-style (e.g., monte carlo or temporal difference) learning. See trainer.py for such methods.',
        allow_abbrev=False
    )

    parser.add_argument(
        '--n-runs',
        type=int,
        help='Number of runs.'
    )

    parser.add_argument(
        '--pdf-save-path',
        type=str,
        help='Path to save PDF to.'
    )

    parser.add_argument(
        '--figure-name',
        type=str,
        help='Name for figure that is generated.'
    )

    parser.add_argument(
        '--environment',
        type=str,
        help='Fully-qualified type name of environment.'
    )

    parser.add_argument(
        '--agent',
        type=str,
        help='Either (1) the fully-qualified type name of agent, or (2) a path to a pickled agent.'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed. Omit to generate an arbitrary random seed.'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot rewards.'
    )

    parsed_args, unparsed_args = parser.parse_known_args(args)

    if parsed_args.random_seed is None:
        random_state = RandomState()
    else:
        random_state = RandomState(parsed_args.random_seed)

    # init environment
    environment_class = load_class(parsed_args.environment)
    environment, unparsed_args = environment_class.init_from_arguments(
        args=unparsed_args,
        random_state=random_state
    )

    # init agent from file if it's a path
    if os.path.exists(os.path.expanduser(parsed_args.agent)):
        with open(os.path.expanduser(parsed_args.agent), 'rb') as f:
            agents = [pickle.load(f)]

    # otherwise, parse arguments for agent (there can't be a policy in this case, as policies only come from prior
    # training/pickling).
    else:
        agent_class = load_class(parsed_args.agent)
        agents, unparsed_args = agent_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            pi=None
        )

    # no unparsed arguments should remain
    if len(unparsed_args) > 0:
        raise ValueError(f'Unparsed arguments:  {unparsed_args}')

    # set up plotting
    pdf = None
    reward_ax = cum_reward_ax = optimal_action_ax = None
    if parsed_args.plot:

        if parsed_args.pdf_save_path:
            pdf = PdfPages(parsed_args.pdf_save_path)

        _, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 9))

        reward_ax = axs[0]
        cum_reward_ax = reward_ax.twinx()
        optimal_action_ax = axs[1]

    # run each agent in the environment
    monitors = []
    for agent in agents:

        print(f'Running {agent} agent in {environment} environment...')

        monitor = Monitor()
        monitors.append(monitor)

        num_runs_per_print = math.ceil(parsed_args.n_runs * 0.05)
        for r in range(parsed_args.n_runs):

            state = environment.reset_for_new_run(agent)
            agent.reset_for_new_run(state)
            monitor.reset_for_new_run()

            environment.run(
                agent=agent,
                monitor=monitor
            )

            num_runs_finished = r + 1
            if (num_runs_finished % num_runs_per_print) == 0:
                percent_done = 100 * (num_runs_finished / parsed_args.n_runs)
                print(f'{percent_done:.0f}% complete (finished {num_runs_finished} of {parsed_args.n_runs} runs)...')

        if parsed_args.plot:

            reward_ax.plot([
                monitor.t_average_reward[t].get_value()
                for t in sorted(monitor.t_average_reward)
            ], linewidth=1, label=agent.name)

            cum_reward_ax.plot([
                monitor.t_average_cumulative_reward[t].get_value()
                for t in sorted(monitor.t_average_cumulative_reward)
            ], linewidth=1, linestyle='--', label=agent.name)

            optimal_action_ax.plot([
                monitor.t_count_optimal_action[t] / parsed_args.n_runs
                for t in sorted(monitor.t_count_optimal_action)
            ], linewidth=1, label=agent.name)

        print()

    # finish plotting
    if parsed_args.plot:

        if parsed_args.figure_name is not None:
            reward_ax.set_title(parsed_args.figure_name)

        reward_ax.set_xlabel('Time step')
        reward_ax.set_ylabel(f'Per-step reward (averaged over {parsed_args.n_runs} run(s))')
        reward_ax.grid()
        reward_ax.legend()
        cum_reward_ax.set_ylabel(f'Cumulative reward (averaged over {parsed_args.n_runs} run(s))')
        cum_reward_ax.legend(loc='lower right')

        optimal_action_ax.set_xlabel('Time step')
        optimal_action_ax.set_ylabel(f'% optimal action selected')
        optimal_action_ax.grid()
        optimal_action_ax.legend()

        if pdf is None:
            plt.show()
        else:
            pdf.savefig()
            pdf.close()

    return monitors


if __name__ == '__main__':
    run(sys.argv[1:])
