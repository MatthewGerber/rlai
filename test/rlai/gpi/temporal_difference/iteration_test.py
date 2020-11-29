import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from test.rlai.utils import get_pi_fixture, get_q_S_A_fixture


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
        num_planning_improvements_per_direct_improvement=None,
        make_final_policy_greedy=False
    )

    pi = get_pi_fixture(mdp_agent.pi)
    q_S_A = get_q_S_A_fixture(q_S_A)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert pi == pi_fixture and q_S_A == q_S_A_fixture


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
        num_planning_improvements_per_direct_improvement=None,
        make_final_policy_greedy=False
    )

    pi = get_pi_fixture(mdp_agent.pi)
    q_S_A = get_q_S_A_fixture(q_S_A)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert pi == pi_fixture and q_S_A == q_S_A_fixture


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
        num_planning_improvements_per_direct_improvement=None,
        make_final_policy_greedy=False
    )

    pi = get_pi_fixture(mdp_agent.pi)
    q_S_A = get_q_S_A_fixture(q_S_A)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert pi == pi_fixture and q_S_A == q_S_A_fixture


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
        num_planning_improvements_per_direct_improvement=None,
        make_final_policy_greedy=False
    )

    pi = get_pi_fixture(mdp_agent.pi)
    q_S_A = get_q_S_A_fixture(q_S_A)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert pi == fixture_pi and q_S_A == fixture_q_S_A
