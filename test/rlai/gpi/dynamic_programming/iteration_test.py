from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld
from rlai.gpi.dynamic_programming.iteration import iterate_policy_v_pi, iterate_policy_q_pi, iterate_value_v_pi, \
    iterate_value_q_pi


def test_policy_iteration():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    # state-value policy iteration
    mdp_agent_v_pi = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent_v_pi.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_policy_v_pi(
        mdp_agent_v_pi,
        mdp_environment,
        0.001,
        True
    )

    # action-value policy iteration
    mdp_agent_q_pi = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent_q_pi.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_policy_q_pi(
        mdp_agent_q_pi,
        mdp_environment,
        0.001,
        True
    )

    # should get the same policy
    assert mdp_agent_v_pi.pi == mdp_agent_q_pi.pi


def test_value_iteration():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    # run policy iteration on v_pi
    mdp_agent_v_pi_policy_iteration = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent_v_pi_policy_iteration.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_policy_v_pi(
        mdp_agent_v_pi_policy_iteration,
        mdp_environment,
        0.001,
        True
    )

    # run value iteration on v_pi
    mdp_agent_v_pi_value_iteration = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent_v_pi_value_iteration.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_value_v_pi(
        mdp_agent_v_pi_value_iteration,
        mdp_environment,
        0.001,
        1,
        True
    )

    assert mdp_agent_v_pi_policy_iteration.pi == mdp_agent_v_pi_value_iteration.pi

    # run value iteration on q_pi
    mdp_agent_q_pi_value_iteration = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent_q_pi_value_iteration.initialize_equiprobable_policy(mdp_environment.SS)

    iterate_value_q_pi(
        mdp_agent_q_pi_value_iteration,
        mdp_environment,
        0.001,
        1,
        True
    )

    assert mdp_agent_q_pi_value_iteration.pi == mdp_agent_v_pi_policy_iteration.pi
