import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld, TrajectorySamplingMdpPlanningEnvironment
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.value_estimation.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.value_estimation.function_approximation.statistical_learning.feature_extraction import StateActionIdentityFeatureExtractor
from rlai.value_estimation.function_approximation.statistical_learning.sklearn import SKLearnSGD
from rlai.value_estimation.tabular import TabularStateActionValueEstimator
from test.rlai.utils import tabular_estimator_legacy_eq, tabular_pi_legacy_eq


def test_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, None)

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
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=0.05,
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

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, None)

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
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=0.05,
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

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, None)

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
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=0.05,
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

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, None)

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
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=1,
        epsilon=0.05,
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

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    epsilon = 0.05

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        epsilon,
        SKLearnSGD(),
        StateActionIdentityFeatureExtractor(),
        f'C(s, levels={[s.i for s in mdp_environment.SS]}):C(a, levels={[a.i for a in mdp_environment.SS[0].AA]})'
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
        num_episodes_per_improvement=10,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=1,
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

    assert mdp_agent.pi == pi_fixture and q_S_A, q_S_A_fixture


def test_expected_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, None)

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
        alpha=0.1,
        mode=Mode.EXPECTED_SARSA,
        n_steps=1,
        epsilon=0.05,
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

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, None)

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
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=3,
        epsilon=0.05,
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
