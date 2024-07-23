import os
import pickle
import shlex
import tempfile
from typing import List

import pytest
from numpy.random import RandomState
from numpy.testing import assert_allclose

from rlai.core import Monitor
from rlai.core.environments.gamblers_problem import GamblersProblem
from rlai.core.environments.mancala import Mancala
from rlai.core.environments.mdp import MdpEnvironment
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator
from rlai.runners.agent_in_environment import run


def test_k_armed_bandit_epsilon_greedy_no_resets():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.2 0.0'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_no_resets.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_no_resets.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_resets_no_alpha():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.2 0.0'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_no_alpha.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_no_alpha.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_resets_with_alpha():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --reset-probability 0.005 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.2 0.0 --alpha 0.1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_with_alpha.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_resets_with_alpha.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_epsilon_reduction():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.2 0.0 --epsilon-reduction-rate 0.01'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_epsilon_reduction.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_epsilon_reduction.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_epsilon_greedy_optimistic():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.0 --initial-q-value 5 --alpha 0.1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_optimistic.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_epsilon_greedy_optimistic.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_upper_confidence_bound():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --agent rlai.core.UpperConfidenceBoundAgent --c 0 1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_upper_confidence_bound.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_upper_confidence_bound.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_preference_gradient_with_baseline():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.core.PreferenceGradientAgent --step-size-alpha 0.1 --use-reward-baseline'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_with_baseline.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_with_baseline.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_k_armed_bandit_preference_gradient_without_baseline():
    """
    Test.
    """

    monitors = run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --q-star-mean 4 --agent rlai.core.PreferenceGradientAgent --step-size-alpha 0.1'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_without_baseline.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_k_armed_bandit_preference_gradient_without_baseline.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_mancala():
    """
    Test.
    """

    monitors = run(shlex.split(f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.mancala.Mancala --initial-count 4 --agent {dump_agent(Mancala(RandomState(1234), None, 4, None))}'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_gamblers_problem():
    """
    Test.
    """

    monitors = run(shlex.split(f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.gamblers_problem.GamblersProblem --p-h 0.4 --agent {dump_agent(GamblersProblem("test", RandomState(1234), None, 0.5))} --log INFO'))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'wb') as file:
    #     pickle.dump(monitors, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'rb') as file:
        monitors_fixture = pickle.load(file)

    assert_monitors(monitors, monitors_fixture)


def test_unparsed_args():
    """
    Test.
    """

    with pytest.raises(ValueError, match='Unparsed arguments'):
        run(shlex.split('--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.2 0.0 --testing'))


def test_plot():
    """
    Test.
    """

    # without pdf (without random seed)
    run(shlex.split('--T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.2 0.0 --plot --figure-name test'))

    # with pdf
    run(shlex.split(f'--random-seed 12345 --T 100 --n-runs 200 --environment rlai.core.environments.bandit.KArmedBandit --k 10 --agent rlai.core.EpsilonGreedyQValueAgent --epsilon 0.2 0.0 --plot --pdf-save-path {tempfile.NamedTemporaryFile(delete=False).name}'))


def dump_agent(
        environment: MdpEnvironment
) -> str:
    """
    Dump agent.

    :param environment: Environment.
    :return: String path.
    """

    # create dummy mdp agent for runner
    # noinspection PyTypeChecker
    stochastic_mdp_agent = ActionValueMdpAgent(
        'foo',
        RandomState(12345),
        1.0,
        TabularStateActionValueEstimator(
            environment,
            None,
            None
        )
    )
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
