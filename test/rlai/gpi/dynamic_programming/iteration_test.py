from numpy.random import RandomState

from rlai.core.environments.gridworld import Gridworld
from rlai.gpi.dynamic_programming.iteration import (
    iterate_policy_v_pi,
    iterate_policy_q_pi,
    iterate_value_v_pi,
    iterate_value_q_pi
)
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator


def test_policy_iteration():
    """
    Test.
    """

    # state-value policy iteration
    random_state = RandomState(12345)
    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
    mdp_agent_v_pi = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )
    iterate_policy_v_pi(
        mdp_agent_v_pi,
        mdp_environment,
        0.001,
        True
    )

    # action-value policy iteration
    random_state = RandomState(12345)
    mdp_environment = Gridworld.example_4_1(random_state, None)
    mdp_agent_q_pi = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )

    iterate_policy_q_pi(
        mdp_agent_q_pi,
        mdp_environment,
        0.001,
        True
    )

    # should get the same policy
    assert mdp_agent_v_pi.pi == mdp_agent_q_pi.pi


def test_value_iteration():
    """
    Test.
    """

    # run policy iteration on v_pi
    random_state = RandomState(12345)
    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
    mdp_agent_v_pi_policy_iteration = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )
    iterate_policy_v_pi(
        mdp_agent_v_pi_policy_iteration,
        mdp_environment,
        0.001,
        True
    )

    # run value iteration on v_pi
    random_state = RandomState(12345)
    mdp_environment = Gridworld.example_4_1(random_state, None)
    mdp_agent_v_pi_value_iteration = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )
    iterate_value_v_pi(
        mdp_agent_v_pi_value_iteration,
        mdp_environment,
        0.001,
        1,
        True
    )

    assert mdp_agent_v_pi_policy_iteration.pi == mdp_agent_v_pi_value_iteration.pi

    # run value iteration on q_pi
    random_state = RandomState(12345)
    mdp_environment = Gridworld.example_4_1(random_state, None)
    mdp_agent_q_pi_value_iteration = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )
    iterate_value_q_pi(
        mdp_agent_q_pi_value_iteration,
        mdp_environment,
        0.001,
        1,
        True
    )

    assert mdp_agent_q_pi_value_iteration.pi == mdp_agent_v_pi_policy_iteration.pi
