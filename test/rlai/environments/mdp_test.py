import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import GamblersProblem
from rlai.gpi.dynamic_programming.iteration import iterate_value_v_pi


def test_gamblers_problem():

    random_state = RandomState(12345)

    mdp_environment: GamblersProblem = GamblersProblem(
        'gamblers problem',
        random_state=random_state,
        T=None,
        p_h=0.4
    )

    mdp_agent_v_pi_value_iteration = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent_v_pi_value_iteration.initialize_equiprobable_policy(mdp_environment.SS)

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
