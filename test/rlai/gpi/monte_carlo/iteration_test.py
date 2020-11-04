import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi


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
        num_improvements=3000,
        num_episodes_per_improvement=1,
        update_upon_every_visit=False,
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


def test_off_policy_monte_carlo():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    # target agent
    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        1
    )
    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    # episode generation (behavior) policy
    off_policy_agent = StochasticMdpAgent(
        'test',
        random_state,
        1
    )
    off_policy_agent.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        update_upon_every_visit=True,
        epsilon=0.0,
        off_policy_agent=off_policy_agent,
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
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_off_policy_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump(pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_off_policy_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert pi == fixture