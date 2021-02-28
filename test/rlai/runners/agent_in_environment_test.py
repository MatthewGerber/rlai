import os
import pickle
import shlex
import tempfile
from typing import Dict, List

import pytest
from numpy.random import RandomState
from numpy.testing import assert_allclose

from rlai.agents.mdp import StochasticMdpAgent
from rlai.policies.tabular import TabularPolicy
from rlai.runners.agent_in_environment import run
from rlai.runners.monitor import Monitor


def test_run():

    # create dummy mdp agent for runner
    stochastic_mdp_agent = StochasticMdpAgent('foo', RandomState(12345), TabularPolicy(None, None), 1.0)
    agent_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(agent_path, 'wb') as f:
        pickle.dump(stochastic_mdp_agent, f)

    run_args_list = [
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --alpha 0.1',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --epsilon-reduction-rate 0.01',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.0 --initial-q-value 5 --alpha 0.1',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.UpperConfidenceBound --c 0 1',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.agents.h_value.PreferenceGradient --step-size-alpha 0.1 --use-reward-baseline',
        '--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.agents.h_value.PreferenceGradient --step-size-alpha 0.1',
        f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.mancala.Mancala --initial-count 4 --agent {agent_path}',
        f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.gamblers_problem.GamblersProblem --p-h 0.4 --agent {agent_path}'
    ]

    run_monitor: Dict[str, List[Monitor]] = dict()

    for run_args in run_args_list:
        run_monitor[run_args] = run(shlex.split(run_args))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/agent_in_environment_test.pickle', 'wb') as file:
    #     pickle.dump(run_monitor, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/agent_in_environment_test.pickle', 'rb') as file:
        run_monitor_fixture = pickle.load(file)

    assert len(run_args_list) == len(run_monitor_fixture.keys())

    for run_args, run_args_fixture in zip(run_args_list, run_monitor_fixture.keys()):

        print(f'Checking test results for run {run_args}...', end='')

        for monitor, monitor_fixture in zip(run_monitor[run_args], run_monitor_fixture[run_args_fixture]):

            assert monitor.cumulative_reward == monitor_fixture.cumulative_reward

            assert_allclose(
                [
                    monitor.t_count_optimal_action[t]
                    for t in sorted(monitor.t_count_optimal_action)
                ],
                [
                    monitor_fixture.t_count_optimal_action[t]
                    for t in sorted(monitor_fixture.t_count_optimal_action)
                ]
            )

            assert_allclose(
                [
                    monitor.t_average_reward[t].get_value()
                    for t in sorted(monitor.t_average_reward)
                ],
                [
                    monitor_fixture.t_average_reward[t].get_value()
                    for t in sorted(monitor_fixture.t_average_reward)
                ]
            )

            assert_allclose(
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


def test_unparsed_args():

    with pytest.raises(ValueError, match='Unparsed arguments'):
        run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --testing'))


def test_plot():

    # without pdf (without random seed)
    run(shlex.split('--T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --plot --figure-name test'))

    # with pdf
    run(shlex.split(f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --plot --pdf-save-path {tempfile.NamedTemporaryFile(delete=False).name}'))
