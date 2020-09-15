import sys
from argparse import ArgumentParser
from typing import List

import matplotlib.pyplot as plt
from numpy.random.mtrand import RandomState

from rl.agents.action import Action
from rl.agents.nonassociative import EpsilonGreedy
from rl.environments.bandit import KArmedBandit
from rl.utils import IncrementalSampleAverager


def k_armed_bandit_with_nonassociative_epsilon_greedy_agent():

    k = 10

    random_state = RandomState(12345)

    bandit = KArmedBandit(
        k=k,
        q_star_mean=0,
        q_star_variance=1,
        reward_variance=1,
        reset_probability=0,
        random_state=random_state
    )

    agent = EpsilonGreedy(
        AA=[Action(i) for i in range(k)],
        epsilon=0.1,
        epsilon_reduction_rate=0,
        random_state=random_state
    )

    t_average_reward = bandit.run(
        agent=agent,
        T=1000,
        n_runs=2000,
        update_ui=plot_t_average_reward
    )

    plot_t_average_reward(t_average_reward)


def plot_t_average_reward(
        t_average_reward: List[IncrementalSampleAverager]
):
    """
    Plot the average reward obtained per tick.

    :param t_average_reward: List of reward averagers.
    """

    plt.plot([averager.get_value() for averager in t_average_reward])
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
