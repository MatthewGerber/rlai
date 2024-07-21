import os
import pickle

import pytest
from numpy.random import RandomState

from rlai.core import Reward, Action, MdpState, Monitor, State
from rlai.core.environments.gamblers_problem import GamblersProblem
from rlai.core.environments.gridworld import Gridworld
from rlai.core.environments.mdp import PrioritizedSweepingMdpPlanningEnvironment, StochasticEnvironmentModel
from rlai.gpi.dynamic_programming.iteration import iterate_value_v_pi
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator
from rlai.utils import sample_list_item


def test_gamblers_problem():
    """
    Test.
    """

    random_state = RandomState(12345)
    mdp_environment: GamblersProblem = GamblersProblem(
        'gamblers problem',
        random_state=random_state,
        T=None,
        p_h=0.4
    )
    mdp_agent_v_pi_value_iteration = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )

    v_pi = iterate_value_v_pi(
        mdp_agent_v_pi_value_iteration,
        mdp_environment,
        0.001,
        1,
        True
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'wb') as file:
    #     pickle.dump(v_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert v_pi == fixture


def test_prioritized_planning_environment():
    """
    Test.
    """

    rng = RandomState(12345)

    planning_environment = PrioritizedSweepingMdpPlanningEnvironment(
        'test',
        rng,
        StochasticEnvironmentModel(),
        1,
        0.3,
        10
    )

    planning_environment.add_state_action_priority(MdpState(1, [], False, False), Action(1), 0.2)
    planning_environment.add_state_action_priority(MdpState(2, [], False, False), Action(2), 0.1)
    planning_environment.add_state_action_priority(MdpState(3, [], False, False), Action(3), 0.3)

    s, a = planning_environment.get_state_action_with_highest_priority()
    assert s is not None and a is not None
    assert s.i == 2 and a.i == 2
    s, a = planning_environment.get_state_action_with_highest_priority()
    assert s is not None and a is not None
    assert s.i == 1 and a.i == 1
    s, a = planning_environment.get_state_action_with_highest_priority()
    assert s is None and a is None


def test_run():
    """
    Test.
    """

    random_state = RandomState(12345)

    mdp_environment: GamblersProblem = GamblersProblem(
        'gamblers problem',
        random_state=random_state,
        T=None,
        p_h=0.4
    )

    agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )

    monitor = Monitor()
    state = mdp_environment.reset_for_new_run(agent)
    agent.reset_for_new_run(state)
    mdp_environment.run(agent, monitor)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_run.pickle', 'wb') as file:
    #     pickle.dump(monitor, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_run.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert monitor.t_average_reward == fixture.t_average_reward


def test_check_marginal_probabilities():
    """
    Test.
    """

    random = RandomState()
    gridworld = Gridworld.example_4_1(random, None)
    gridworld.p_S_prime_R_given_S_A[gridworld.SS[0]][gridworld.a_left][gridworld.SS[0]][Reward(1, -1)] = 1.0

    with pytest.raises(ValueError, match='Expected next-state/next-reward marginal probability of 1.0, but got 2.0'):
        gridworld.check_marginal_probabilities()


def test_stochastic_environment_model():
    """
    Test.
    """

    random_state = RandomState(12345)

    model = StochasticEnvironmentModel()

    actions = [
        Action(i)
        for i in range(5)
    ]

    states = [
        State(i, actions)
        for i in range(5)
    ]

    for t in range(1000):
        state = sample_list_item(states, None, random_state)
        action = sample_list_item(state.AA, None, random_state)
        next_state = sample_list_item(states, None, random_state)
        reward = Reward(None, random_state.randint(10))
        model.update(state, action, next_state, reward)

    environment_sequence = []
    for i in range(1000):
        state = model.sample_state(random_state)
        action = model.sample_action(state, random_state)
        next_state, reward_value = model.sample_next_state_and_reward(state, action, random_state)
        environment_sequence.append((next_state, reward_value))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_stochastic_environment_model.pickle', 'wb') as file:
    #     pickle.dump(environment_sequence, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_stochastic_environment_model.pickle', 'rb') as file:
        environment_sequence_fixture = pickle.load(file)

    assert environment_sequence == environment_sequence_fixture
