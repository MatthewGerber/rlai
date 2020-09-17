import sys
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.nonassociative import EpsilonGreedy
from rl.environments.bandit import KArmedBandit
from rl.runners.monitor import Monitor


def k_armed_bandit_with_nonassociative_epsilon_greedy_agent(
        args: List[str],
        output_name: str
):
    """
    Run a k-armed bandit environment with a nonassociative, epsilon-greedy agent.

    :param args: Arguments.
    :param output_name: Output name.
    """

    parser = ArgumentParser(description='Run a k-armed bandit environment with a nonassociative, epsilon-greedy agent.')

    parser.add_argument(
        '--epsilons',
        type=str,
        default='0',
        help='Comma-separated list of epsilon values to evaluate.'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=None,
        help='Step-size to use in incremental reward averaging. Pass None for decreasing (i.e., unweighted average) or a constant in (0, 1] for recency weighted.'
    )

    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='Number of bandit arms.'
    )

    parser.add_argument(
        '--T',
        type=int,
        default=1000,
        help='Number of time steps.'
    )

    parser.add_argument(
        '--n-runs',
        type=int,
        default=2000,
        help='Number of runs.'
    )

    parser.add_argument(
        '--reset-probability',
        type=float,
        default=0.0,
        help="Probability of resetting the bandit's arms at each time step. This effectively creates a nonstationary environment."
    )

    args = parser.parse_args(args)

    epsilons = [
        float(epsilon_str)
        for epsilon_str in args.epsilons.split(',')
    ]

    random_state = RandomState(12345)

    agents = [
        EpsilonGreedy(
            AA=[Action(i) for i in range(args.k)],
            epsilon=epsilon,
            epsilon_reduction_rate=0,
            random_state=random_state,
            alpha=args.alpha,
            name=f'epsilon-greedy (e={epsilon:0.2f})'
        )
        for epsilon in epsilons
    ]

    bandits = [
        KArmedBandit(
            name=f'{args.k}-armed bandit',
            k=args.k,
            q_star_mean=0,
            q_star_variance=1,
            reward_variance=1,
            reset_probability=args.reset_probability,
            random_state=random_state
        )
        for _ in range(len(agents))
    ]

    monitor = Monitor(
        T=args.T
    )

    fig, axs = plt.subplots(2, 1, sharex='all', figsize=(6, 9))

    reward_ax = axs[0]
    optimal_action_ax = axs[1]

    for agent, bandit in zip(agents, bandits):

        print(f'Running {agent} in {bandit}...')

        monitor.reset()

        for run in range(args.n_runs):

            agent.reset()
            bandit.reset()

            bandit.run(
                agent=agent,
                T=args.T,
                monitor=monitor
            )

            runs_finished = run + 1
            if (runs_finished % 100) == 0:
                percent_done = 100 * (runs_finished / args.n_runs)
                print(f'{percent_done:.0f}% complete (finished {runs_finished} of {args.n_runs} runs)...')

        reward_ax.plot([
            averager.get_value()
            for averager in monitor.t_average_reward
        ], label=agent.name)

        optimal_action_ax.plot([
            count_optimal_action / args.n_runs
            for count_optimal_action in monitor.t_count_optimal_action
        ], label=agent.name)

        print()

    reward_ax.set_title(output_name)
    reward_ax.set_xlabel('Time step')
    reward_ax.set_ylabel(f'Per-step reward (averaged over {args.n_runs} runs)')
    reward_ax.grid()
    reward_ax.legend()

    optimal_action_ax.set_xlabel('Time step')
    optimal_action_ax.set_ylabel(f'% optimal action selected')
    optimal_action_ax.grid()
    optimal_action_ax.legend()

    plt.show()


def main(
        argv: List[str]
):

    parser = ArgumentParser(description='Run an experiment.')

    parser.add_argument(
        'runner_name',
        type=str,
        help='Runner name.'
    )

    parser.add_argument(
        '--output-name',
        type=str,
        default='_'.join(argv),
        help='Name of output.'
    )

    args, unknown_args = parser.parse_known_args(argv)

    runner = globals()[args.runner_name]

    runner(
        args=unknown_args,
        output_name=args.output_name
    )


if __name__ == '__main__':
    main(sys.argv[1:])
