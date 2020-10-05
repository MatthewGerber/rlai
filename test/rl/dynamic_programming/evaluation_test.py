import os
import pickle

import numpy as np
from numpy.random import RandomState

from rl.agents.mdp import Stochastic
from rl.dynamic_programming.policy_iteration import evaluate_v, evaluate_q, iterate_policy_v
from rl.environments.mdp import Gridworld


def test_evaluate_v():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    mdp_agent = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    state_value = evaluate_v(
        mdp_agent,
        mdp_environment,
        0.001,
        True
    )

    state_value_not_in_place = evaluate_v(
        mdp_agent,
        mdp_environment,
        0.001,
        False
    )

    assert list(state_value.keys()) == list(state_value_not_in_place.keys())

    assert np.allclose(list(state_value.values()), list(state_value_not_in_place.values()), atol=0.01)

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    state_value = {
        s.i: state_value[s]
        for s in state_value
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_state_value.pickle', 'wb') as file:
    #     pickle.dump(state_value, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_state_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert state_value == fixture


def test_evaluate_q():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    mdp_agent = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    action_value = evaluate_q(
        mdp_agent,
        mdp_environment,
        0.001,
        True
    )

    action_value_not_in_place = evaluate_q(
        mdp_agent,
        mdp_environment,
        0.001,
        False
    )

    assert list(action_value.keys()) == list(action_value_not_in_place.keys())

    for s in action_value:
        for a in action_value[s]:
            assert np.allclose(action_value[s][a], action_value_not_in_place[s][a], atol=0.01)

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    action_value = {
        s.i: {
            a.i: action_value[s][a]
            for a in action_value[s]
        }
        for s in action_value
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_action_value.pickle', 'wb') as file:
    #     pickle.dump(action_value, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_action_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert action_value == fixture


def test_policy_iteration():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    mdp_agent = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    iterate_policy_v(
        mdp_agent,
        mdp_environment,
        0.001,
        True
    )
    
    assert False
