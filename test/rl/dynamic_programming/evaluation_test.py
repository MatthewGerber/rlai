import os
import pickle

import numpy as np
from numpy.random import RandomState

from rl.agents.mdp import Stochastic
from rl.gpi.dynamic_programming.evaluation import evaluate_v_pi, evaluate_q_pi
from rl.gpi.dynamic_programming.iteration import iterate_value_v_pi, iterate_value_q_pi, iterate_policy_v_pi, iterate_policy_q_pi
from rl.environments.mdp import Gridworld


def test_evaluate_v_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    v_pi, _ = evaluate_v_pi(
        agent=mdp_agent,
        theta=0.001,
        num_iterations=None,
        update_in_place=True
    )

    v_pi_not_in_place, _ = evaluate_v_pi(
        agent=mdp_agent,
        theta=0.001,
        num_iterations=None,
        update_in_place=False
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

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    q_pi, _ = evaluate_q_pi(
        agent=mdp_agent,
        theta=0.001,
        num_iterations=None,
        update_in_place=True
    )

    q_pi_not_in_place, _ = evaluate_q_pi(
        agent=mdp_agent,
        theta=0.001,
        num_iterations=None,
        update_in_place=False
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

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

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
        0.001,
        True
    )

    # should get the same policy
    assert mdp_agent_v_pi.pi == mdp_agent_q_pi.pi


def test_value_iteration():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    # run policy iteration on v_pi
    mdp_agent_v_pi_policy_iteration = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    iterate_policy_v_pi(
        mdp_agent_v_pi_policy_iteration,
        0.001,
        True
    )

    # run value iteration on v_pi
    mdp_agent_v_pi_value_iteration = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    iterate_value_v_pi(
        mdp_agent_v_pi_value_iteration,
        0.001,
        1,
        True
    )

    assert mdp_agent_v_pi_policy_iteration.pi == mdp_agent_v_pi_value_iteration.pi

    # run value iteration on q_pi
    mdp_agent_q_pi_value_iteration = Stochastic(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    iterate_value_q_pi(
        mdp_agent_q_pi_value_iteration,
        0.001,
        1,
        True
    )

    assert mdp_agent_q_pi_value_iteration.pi == mdp_agent_v_pi_policy_iteration.pi
