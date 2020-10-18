import os
import pickle

from numpy.random import RandomState

from rl.agents.mdp import StochasticMdpAgent
from rl.gpi.dynamic_programming.iteration import iterate_value_v_pi
from rl.environments.mdp import GamblersProblem


def test_gamblers_problem():

    random_state = RandomState(12345)

    mdp_environment: GamblersProblem = GamblersProblem(
        'gamblers problems',
        random_state=random_state,
        p_h=0.4
    )

    mdp_agent_v_pi_value_iteration = StochasticMdpAgent(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    v_pi = iterate_value_v_pi(
        mdp_agent_v_pi_value_iteration,
        0.001,
        1,
        True
    )

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    v_pi = {
        s.i: v_pi[s]
        for s in v_pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'wb') as file:
    #     pickle.dump(v_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert v_pi == fixture
