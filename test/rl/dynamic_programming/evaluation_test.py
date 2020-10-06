import os
import pickle

import numpy as np
from numpy.random import RandomState

from rl.agents.mdp import Stochastic
from rl.dynamic_programming.policy_iteration import evaluate_v_pi, evaluate_q_pi, iterate_policy_v_pi, \
    iterate_policy_q_pi
from rl.environments.mdp import Gridworld


def test_evaluate_v_pi():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    mdp_agent = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    v_pi = evaluate_v_pi(
        mdp_agent,
        mdp_environment,
        0.001,
        True
    )

    v_pi_not_in_place = evaluate_v_pi(
        mdp_agent,
        mdp_environment,
        0.001,
        False
    )

    assert list(v_pi.keys()) == list(v_pi_not_in_place.keys())

    assert np.allclose(list(v_pi.values()), list(v_pi_not_in_place.values()), atol=0.01)

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    v_pi = {
        s.i: v_pi[s]
        for s in v_pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_state_value.pickle', 'wb') as file:
    #     pickle.dump(v_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_state_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert v_pi == fixture


def test_evaluate_q_pi():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    mdp_agent = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    q_pi = evaluate_q_pi(
        mdp_agent,
        mdp_environment,
        0.001,
        True
    )

    q_pi_not_in_place = evaluate_q_pi(
        mdp_agent,
        mdp_environment,
        0.001,
        False
    )

    assert list(q_pi.keys()) == list(q_pi_not_in_place.keys())

    for s in q_pi:
        for a in q_pi[s]:
            assert np.allclose(q_pi[s][a], q_pi_not_in_place[s][a], atol=0.01)

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    q_pi = {
        s.i: {
            a.i: q_pi[s][a]
            for a in q_pi[s]
        }
        for s in q_pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_action_value.pickle', 'wb') as file:
    #     pickle.dump(q_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_action_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert q_pi == fixture


def test_policy_iteration():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    # state-value policy iteration
    mdp_agent_v_pi = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    iterate_policy_v_pi(
        mdp_agent_v_pi,
        mdp_environment,
        0.001,
        True
    )

    # action-value policy iteration
    mdp_agent_q_pi = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    iterate_policy_q_pi(
        mdp_agent_q_pi,
        mdp_environment,
        0.001,
        True
    )

    # should get the same policy
    assert mdp_agent_v_pi.pi == mdp_agent_q_pi.pi
