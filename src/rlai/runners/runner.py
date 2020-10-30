import importlib
import os
import sys
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.agents.mdp import MdpAgent
from rlai.runners import FIGURES_DIRECTORY
from rlai.runners.monitor import Monitor


def load_class(
        fully_qualified_class_name: str
):
    """
    Load class from its fully-qualified name (e.g., xxx.yyy.Class).

    :param fully_qualified_class_name: Name.
    :return: Class reference.
    """

    (module_name, fully_qualified_class_name) = fully_qualified_class_name.rsplit('.', 1)
    module_ref = importlib.import_module(module_name)
    class_ref = getattr(module_ref, fully_qualified_class_name)

    return class_ref


def run(
        args: List[str]
):
    random_state = RandomState(12345)

    parser = ArgumentParser(description='Run an agent within an environment.', allow_abbrev=False)

    parser.add_argument(
        '--n-runs',
        type=int,
        default=2000,
        help='Number of runs.'
    )

    parser.add_argument(
        '--T',
        type=int,
        default=1000,
        help='Number of time steps per run.'
    )

    parser.add_argument(
        '--figure-name',
        type=str,
        default='_'.join(args),
        help='Name of output figure.'
    )

    parser.add_argument(
        '--save-to-pdf',
        action='store_true',
        help='Whether or not to save the output figure to disk as a PDF.'
    )

    parser.add_argument(
        '--environment',
        type=str,
        default='rlai.environments.bandit.KArmedBandit',
        help='Fully-qualified class name of environment.'
    )

    parser.add_argument(
        '--agent',
        type=str,
        default='rlai.agents.greedy.EpsilonGreedy',
        help='Fully-qualified class name of agent.'
    )

    parsed_args, unparsed_args = parser.parse_known_args(args)

    # init environment
    environment_class = load_class(parsed_args.environment)
    environment, unparsed_args = environment_class.init_from_arguments(
        args=unparsed_args,
        random_state=random_state
    )

    # init agent(s)
    agent_class = load_class(parsed_args.agent)
    agents, unparsed_args = agent_class.init_from_arguments(
        args=unparsed_args,
        random_state=random_state
    )

    # no unparsed arguments should remain
    if len(unparsed_args) > 0:
        raise ValueError(f'Unparsed arguments:  {unparsed_args}')

    # if we're running mdp agents, have each agent solve the mdp with the specified function/parameters.
    for agent in agents:
        if isinstance(agent, MdpAgent):
            agent.initialize_equiprobable_policy(environment.SS)
            agent.solve_mdp()

    # set up plot
    pdf = None
    if parsed_args.save_to_pdf:
        pdf = PdfPages(os.path.join(FIGURES_DIRECTORY, parsed_args.figure_name + '.pdf'))

    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 9))

    reward_ax = axs[0]
    cum_reward_ax = reward_ax.twinx()
    optimal_action_ax = axs[1]

    # run each agent in the environment
    monitors = []
    for agent in agents:

        print(f'Running {agent} agent in {environment} environment...')

        monitor = Monitor(
            T=parsed_args.T
        )

        monitors.append(monitor)

        num_runs_per_print = int(parsed_args.n_runs * 0.05)
        for r in range(parsed_args.n_runs):

            state = environment.reset_for_new_run()
            agent.reset_for_new_run(state)
            monitor.reset_for_new_run()

            environment.run(
                agent=agent,
                T=parsed_args.T,
                monitor=monitor
            )

            num_runs_finished = r + 1
            if (num_runs_finished % num_runs_per_print) == 0:
                percent_done = 100 * (num_runs_finished / parsed_args.n_runs)
                print(f'{percent_done:.0f}% complete (finished {num_runs_finished} of {parsed_args.n_runs} runs)...')

        reward_ax.plot([
            averager.get_value()
            for averager in monitor.t_average_reward
        ], linewidth=1, label=agent.name)

        cum_reward_ax.plot([
            averager.get_value()
            for averager in monitor.t_average_cumulative_reward
        ], linewidth=1, linestyle='--', label=agent.name)

        optimal_action_ax.plot([
            count_optimal_action / parsed_args.n_runs
            for count_optimal_action in monitor.t_count_optimal_action
        ], linewidth=1, label=agent.name)

        print()

    reward_ax.set_title(parsed_args.figure_name)
    reward_ax.set_xlabel('Time step')
    reward_ax.set_ylabel(f'Per-step reward (averaged over {parsed_args.n_runs} runs)')
    reward_ax.grid()
    reward_ax.legend()
    cum_reward_ax.set_ylabel(f'Cumulative reward (averaged over {parsed_args.n_runs} runs)')
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
