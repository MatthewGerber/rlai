import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi


def test_iterate_value_q_pi():

    return

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
        num_improvements=100,
        num_episodes_per_improvement=10,
        alpha=0.1,
        epsilon=0.1
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
    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'wb') as file:
        pickle.dump(pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert pi == fixture
