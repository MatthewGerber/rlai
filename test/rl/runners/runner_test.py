import os
import pickle
from typing import Dict, List

from numpy.testing import assert_array_equal

from rl.runners.monitor import Monitor
from rl.runners.runner import run


def test_run():

    run_args_list = [
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --agent rl.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0',
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rl.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0',
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rl.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --alpha 0.1',
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --agent rl.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --epsilon-reduction-rate 0.01',
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --agent rl.agents.q_value.EpsilonGreedy --epsilon 0.0 --initial-q-value 5 --alpha 0.1',
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --agent rl.agents.q_value.UpperConfidenceBound --c 0 1',
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rl.agents.h_value.PreferenceGradient --step-size-alpha 0.1 --use-reward-baseline',
        '--T 100 --n-runs 200 --environment rl.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rl.agents.h_value.PreferenceGradient --step-size-alpha 0.1',
        '--T 50 --n-runs 5000 --environment rl.environments.mdp.GamblersProblem --p-h 0.4 --agent rl.agents.mdp.StochasticMdpAgent --gamma 1 --mdp-solver rl.gpi.dynamic_programming.iteration.iterate_value_v_pi --theta 0.001 --update-in-place --evaluation-iterations-per-improvement 1',
        '--T 30 --n-runs 500 --environment rl.environments.mdp.Gridworld --id example_4_1 --agent rl.agents.mdp.StochasticMdpAgent --gamma 1 --mdp-solver rl.gpi.dynamic_programming.iteration.iterate_value_v_pi --theta 0.001 --update-in-place --evaluation-iterations-per-improvement 1'
    ]

    run_monitor: Dict[str, List[Monitor]] = dict()

    for run_args in run_args_list:
        run_monitor[run_args] = run(run_args.split())

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_run.pickle', 'wb') as file:
    #     pickle.dump(run_monitor, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_run.pickle', 'rb') as file:
        run_monitor_fixture = pickle.load(file)

    for run_args, run_args_fixture in zip(run_args_list, run_monitor_fixture.keys()):
        print(f'Checking test results for run {run_args}...', end='')
        for monitor, monitor_fixture in zip(run_monitor[run_args], run_monitor_fixture[run_args_fixture]):

            assert monitor.cumulative_reward == monitor_fixture.cumulative_reward

            assert_array_equal(monitor.t_count_optimal_action, monitor_fixture.t_count_optimal_action)

            assert_array_equal(
                [r.get_value() for r in monitor.t_average_reward],
                [r.get_value() for r in monitor_fixture.t_average_reward]
            )

            assert_array_equal(
                [r.get_value() for r in monitor.t_average_cumulative_reward],
                [r.get_value() for r in monitor_fixture.t_average_cumulative_reward]
            )

        print('passed.')
