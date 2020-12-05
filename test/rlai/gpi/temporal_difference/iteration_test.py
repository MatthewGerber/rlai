import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld, TrajectorySamplingMdpPlanningEnvironment
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.planning.environment_models import StochasticEnvironmentModel


def test_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A = iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=0.05,
        planning_environment=None,
        make_final_policy_greedy=False
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert mdp_agent.pi == pi_fixture and q_S_A == q_S_A_fixture


def test_sarsa_iterate_value_q_pi_make_greedy():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A = iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=0.05,
        planning_environment=None,
        make_final_policy_greedy=True
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_make_greedy.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_make_greedy.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert mdp_agent.pi == pi_fixture and q_S_A == q_S_A_fixture


def test_sarsa_iterate_value_q_pi_with_trajectory_planning():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    planning_environment = TrajectorySamplingMdpPlanningEnvironment(
        'test planning',
        random_state,
        StochasticEnvironmentModel(),
        10,
        None
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A = iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=0.05,
        planning_environment=planning_environment,
        num_improvements_per_plot=100,
        make_final_policy_greedy=True
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_planning.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_planning.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert mdp_agent.pi == pi_fixture and q_S_A == q_S_A_fixture


def test_q_learning_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A = iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=1,
        epsilon=0.05,
        planning_environment=None,
        make_final_policy_greedy=False
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert mdp_agent.pi == pi_fixture and q_S_A == q_S_A_fixture


def test_expected_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A = iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.EXPECTED_SARSA,
        n_steps=1,
        epsilon=0.05,
        planning_environment=None,
        make_final_policy_greedy=False
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert mdp_agent.pi == pi_fixture and q_S_A == q_S_A_fixture


def test_n_step_q_learning_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A = iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=3,
        epsilon=0.05,
        planning_environment=None,
        make_final_policy_greedy=False
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert mdp_agent.pi == fixture_pi and q_S_A == fixture_q_S_A
