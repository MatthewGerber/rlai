import os
import pickle

from numpy.random import RandomState

from rl.agents.mdp import StochasticMdpAgent
from rl.environments.mdp import Gridworld
from rl.gpi.monte_carlo.evaluation import evaluate_v_pi, evaluate_q_pi
from rl.gpi.monte_carlo.iteration import iterate_value_q_pi


def test_evaluate_v_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        1,
        None,
        None
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    v_pi = evaluate_v_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_episodes=1000
    )

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    v_pi = {
        s.i: v_pi[s]
        for s in v_pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_value.pickle', 'wb') as file:
    #     pickle.dump(v_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert v_pi == fixture


def test_evaluate_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        1,
        None,
        None
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A, evaluated_states, _ = evaluate_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_episodes=1000,
        exploring_starts=True
    )

    assert len(q_S_A) == len(evaluated_states)
    assert all(s in q_S_A for s in evaluated_states)

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    q_pi = {
        s.i: {
            a: q_S_A[s][a].get_value()
            for a in q_S_A[s]
        }
        for s in q_S_A
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_action_value.pickle', 'wb') as file:
    #     pickle.dump(q_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_action_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert q_pi == fixture


def test_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        1,
        None,
        None
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=3000,
        num_episodes_per_improvement=1,
        epsilon=0.1,
        num_improvements_per_plot=None
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
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump(pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert pi == fixture
