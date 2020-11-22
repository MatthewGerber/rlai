import os
import pickle
from typing import Dict, List

from numpy.testing import assert_array_equal

from rlai.runners.agent_in_environment import run
from rlai.runners.monitor import Monitor


def test_run():

    run_args_list = [
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --alpha 0.1',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --epsilon-reduction-rate 0.01',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.0 --initial-q-value 5 --alpha 0.1',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.UpperConfidenceBound --c 0 1',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.agents.h_value.PreferenceGradient --step-size-alpha 0.1 --use-reward-baseline',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.agents.h_value.PreferenceGradient --step-size-alpha 0.1'
    ]

    run_monitor: Dict[str, List[Monitor]] = dict()

    for run_args in run_args_list:
        run_monitor[run_args] = run(run_args.split())

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/agent_in_environment_test.pickle', 'wb') as file:
    #     pickle.dump(run_monitor, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/agent_in_environment_test.pickle', 'rb') as file:
        run_monitor_fixture = pickle.load(file)

    for run_args, run_args_fixture in zip(run_args_list, run_monitor_fixture.keys()):

        print(f'Checking test results for run {run_args}...', end='')

        for monitor, monitor_fixture in zip(run_monitor[run_args], run_monitor_fixture[run_args_fixture]):

            assert monitor.cumulative_reward == monitor_fixture.cumulative_reward

            assert_array_equal(
                [
                    monitor.t_count_optimal_action[t]
                    for t in sorted(monitor.t_count_optimal_action)
                ],
                [
                    monitor_fixture.t_count_optimal_action[t]
                    for t in sorted(monitor_fixture.t_count_optimal_action)
                ]
            )

            assert_array_equal(
                [
                    monitor.t_average_reward[t].get_value()
                    for t in sorted(monitor.t_average_reward)
                ],
                [
                    monitor_fixture.t_average_reward[t].get_value()
                    for t in sorted(monitor_fixture.t_average_reward)
                ]
            )

            assert_array_equal(
                [
                    monitor.t_average_cumulative_reward[t].get_value()
                    for t in sorted(monitor.t_average_cumulative_reward)
                ],
                [
                    monitor_fixture.t_average_cumulative_reward[t].get_value()
                    for t in sorted(monitor_fixture.t_average_cumulative_reward)
                ]
            )

        print('passed.')
