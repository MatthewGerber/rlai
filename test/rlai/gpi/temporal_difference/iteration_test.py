import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld, GamblersProblem
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi


def test_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.SARSA,
        epsilon=0.05
    )

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    pi = {
        s.i: {
            a: mdp_agent.pi[s][a]
            for a in mdp_agent.pi[s]
        }
        for s in mdp_agent.pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump(pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert pi == fixture


def test_q_learning_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        epsilon=0.05
    )

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    pi = {
        s.i: {
            a: mdp_agent.pi[s][a]
            for a in mdp_agent.pi[s]
        }
        for s in mdp_agent.pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump(pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert pi == fixture


def test_expected_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.EXPECTED_SARSA,
        epsilon=0.05
    )

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    pi = {
        s.i: {
            a: mdp_agent.pi[s][a]
            for a in mdp_agent.pi[s]
        }
        for s in mdp_agent.pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump(pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert pi == fixture
