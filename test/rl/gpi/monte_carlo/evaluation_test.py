import os
import pickle

from numpy.random import RandomState

from rl.agents.mdp import StochasticMdpAgent
from rl.environments.mdp import Gridworld
from rl.gpi.monte_carlo.evaluation import evaluate_v_pi, evaluate_q_pi


def test_evaluate_v_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

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
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    q_pi = evaluate_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_episodes=1000
    )

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    q_pi = {
        s.i: {
            a: q_pi[s][a]
            for a in q_pi[s]
        }
        for s in q_pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_action_value.pickle', 'wb') as file:
    #     pickle.dump(q_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_action_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert q_pi == fixture
