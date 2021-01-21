import os
import pickle

import numpy as np
from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import TrajectorySamplingMdpPlanningEnvironment
from rlai.environments.gridworld import Gridworld, GridworldFeatureExtractor
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.value_estimation.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.value_estimation.function_approximation.models.feature_extraction import (
    StateActionIdentityFeatureExtractor
)
from rlai.value_estimation.function_approximation.models.sklearn import SKLearnSGD
from rlai.value_estimation.tabular import TabularStateActionValueEstimator
from test.rlai.utils import tabular_estimator_legacy_eq, tabular_pi_legacy_eq


def test_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_sarsa_iterate_value_q_pi_make_greedy():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_make_greedy.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_make_greedy.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_sarsa_iterate_value_q_pi_with_trajectory_planning():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    planning_environment = TrajectorySamplingMdpPlanningEnvironment(
        'test planning',
        random_state,
        StochasticEnvironmentModel(),
        10,
        None
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=epsilon,
        planning_environment=planning_environment,
        make_final_policy_greedy=True,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_planning.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_planning.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_q_learning_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=1,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_q_learning_iterate_value_q_pi_function_approximation():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    epsilon = 0.05

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        epsilon,
        SKLearnSGD(random_state=random_state),
        StateActionIdentityFeatureExtractor(mdp_environment),
        f'C(s, levels={[s.i for s in mdp_environment.SS]}):C(a, levels={[a.i for a in mdp_environment.SS[0].AA]})',
        False,
        None,
        None
    )

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=5,
        num_episodes_per_improvement=5,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert np.allclose(mdp_agent.pi.estimator.model.model.coef_, pi_fixture.estimator.model.model.coef_)


def test_q_learning_iterate_value_q_pi_function_approximation_no_formula():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    epsilon = 0.05

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        epsilon,
        SKLearnSGD(random_state=random_state),
        GridworldFeatureExtractor(mdp_environment),
        None,
        False,
        None,
        None
    )

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=20,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation_no_formula.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation_no_formula.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert np.allclose(mdp_agent.pi.estimator.model.model.coef_, pi_fixture.estimator.model.model.coef_)
    assert mdp_agent.pi.format_state_action_probs(mdp_environment.SS) == pi_fixture.format_state_action_probs(mdp_environment.SS)


def test_expected_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.EXPECTED_SARSA,
        n_steps=1,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_n_step_q_learning_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=3,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, fixture_pi) and tabular_estimator_legacy_eq(q_S_A, fixture_q_S_A)
