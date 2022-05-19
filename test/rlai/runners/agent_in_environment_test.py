import os
import pickle
import shlex
import tempfile
from typing import List

import pytest
from numpy.random import RandomState
from numpy.testing import assert_allclose

from rlai.agents.mdp import StochasticMdpAgent, ActionValueMdpAgent
from rlai.policies.tabular import TabularPolicy
from rlai.runners.agent_in_environment import run
from rlai.runners.monitor import Monitor


def test_k_armed_bandit_epsilon_greedy_no_resets():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_no_resets.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_no_resets.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_resets_no_alpha():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_no_alpha.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_no_alpha.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_resets_with_alpha():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --alpha 0.1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_with_alpha.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_with_alpha.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_epsilon_reduction():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --epsilon-reduction-rate 0.01'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_epsilon_reduction.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_epsilon_reduction.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_optimistic():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.0 --initial-q-value 5 --alpha 0.1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_optimistic.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_optimistic.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_upper_confidence_bound():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.UpperConfidenceBound --c 0 1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_upper_confidence_bound.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_upper_confidence_bound.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_preference_gradient_with_baseline():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.agents.h_value.PreferenceGradient --step-size-alpha 0.1 --use-reward-baseline'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_with_baseline.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_with_baseline.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_preference_gradient_without_baseline():

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.agents.h_value.PreferenceGradient --step-size-alpha 0.1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_without_baseline.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_without_baseline.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_mancala():

    monitors = run(shlex.split(f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.mancala.Mancala --initial-count 4 --agent {dump_agent()}'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_gamblers_problem():

    monitors = run(shlex.split(f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.gamblers_problem.GamblersProblem --p-h 0.4 --agent {dump_agent()} --log INFO'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_unparsed_args():

    with pytest.raises(ValueError, match='Unparsed arguments'):
        run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --testing'))


def test_plot():

    # without pdf (without random seed)
    run(shlex.split('--T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --plot --figure-name test'))

    # with pdf
    run(shlex.split(f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.environments.bandit.KArmedBandit --k 10 --agent rlai.agents.q_value.EpsilonGreedy --epsilon 0.2 0.0 --plot --pdf-save-path {tempfile.NamedTemporaryFile(delete=False).name}'))


def dump_agent() -> str:

    # create dummy mdp agent for runner
    stochastic_mdp_agent = ActionValueMdpAgent('foo', RandomState(12345), 1.0, )
    agent_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(agent_path, 'wb') as f:
        pickle.dump(stochastic_mdp_agent, f)

    return agent_path


def assert_monitors(
        monitors: List[Monitor],
        monitors_fixture: List[Monitor]
):
    """
    Assert test results for a list of monitors and their fixtures.

    :param monitors: Monitors.
    :param monitors_fixture: Monitors fixture.
    """

    assert len(monitors) == len(monitors_fixture)

    for monitor, monitor_fixture in zip(monitors, monitors_fixture):

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
