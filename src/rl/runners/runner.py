import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from numpy.random import RandomState

from rl.agents.action import Action
from rl.agents.nonassociative import EpsilonGreedy
from rl.environments.bandit import KArmedBandit
from rl.runners.monitor import Monitor


def k_armed_bandit_with_nonassociative_epsilon_greedy_agent():

    k = 10
    T = 1000
    n_runs = 2000

    random_state = RandomState(12345)

    agents = [
        EpsilonGreedy(
            AA=[Action(i) for i in range(k)],
            epsilon=epsilon,
            epsilon_reduction_rate=0,
            random_state=random_state,
            name=f'Epsilon-greedy (e={epsilon:0.5f})'
        )
        for epsilon in [0.1, 0.01, 0]
    ]

    bandits = [
        KArmedBandit(
            k=k,
            q_star_mean=0,
            q_star_variance=1,
            reward_variance=1,
            reset_probability=0,
            random_state=random_state,
            name=f'{k}-armed bandit'
        )
        for _ in range(len(agents))
    ]

    monitor = Monitor(
        T=T
    )

    for agent, bandit in zip(agents, bandits):

        print(f'Running {agent} in {bandit}...')

        monitor.reset()

        for run in range(n_runs):

            agent.reset()
            bandit.reset()

            bandit.run(
                agent=agent,
                T=T,
                monitor=monitor
            )

            runs_finished = run + 1
            if (runs_finished % 100) == 0:
                percent_done = 100 * (runs_finished / n_runs)
                print(f'{percent_done:.0f}% complete (finished {runs_finished} of {n_runs} runs)...')

        plt.plot([averager.get_value() for averager in monitor.t_average_reward], label=agent.name)

        print()

    plt.grid()
    plt.legend()
    plt.show()


def parse_arguments(
        arguments
):
    parser = ArgumentParser(description='Run a scenario.')

    parser.add_argument(
        'scenario_name',
        type=str,
        help='Scenario name.'
    )

    return parser.parse_args(arguments)


def main(argv):

    args = parse_arguments(argv)

    scenario = globals()[args.scenario_name]

    scenario()


if __name__ == '__main__':
    main(sys.argv[1:])
