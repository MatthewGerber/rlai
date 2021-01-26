import os
import pickle
import tempfile

import pytest
from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.gridworld import Gridworld
from rlai.environments.mdp import TrajectorySamplingMdpPlanningEnvironment
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.value_estimation.tabular import TabularStateActionValueEstimator
from test.rlai.utils import tabular_estimator_legacy_eq, tabular_pi_legacy_eq


def test_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.1

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=3000,
        num_episodes_per_improvement=1,
        update_upon_every_visit=False,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_off_policy_monte_carlo():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.0

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    # target agent
    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    # episode generation (behavior) policy
    off_policy_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        update_upon_every_visit=True,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A,
        off_policy_agent=off_policy_agent
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_off_policy_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_off_policy_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_invalid_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.0

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    # target agent
    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    # episode generation (behavior) policy
    off_policy_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    with pytest.raises(ValueError, match='Planning environments are not currently supported for Monte Carlo iteration.'):
        iterate_value_q_pi(
            agent=mdp_agent,
            environment=mdp_environment,
            num_improvements=100,
            num_episodes_per_improvement=1,
            update_upon_every_visit=True,
            epsilon=epsilon,
            planning_environment=TrajectorySamplingMdpPlanningEnvironment('foo', random_state, StochasticEnvironmentModel(), 100, None),
            make_final_policy_greedy=False,
            q_S_A=q_S_A,
            off_policy_agent=off_policy_agent
        )

    # test warning...no off-policy agent with epsilon=0.0
    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        update_upon_every_visit=True,
        epsilon=0.0,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A,
        off_policy_agent=None
    )


def test_iterate_value_q_pi_with_pdf():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.1

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=3000,
        num_episodes_per_improvement=1,
        update_upon_every_visit=False,
        epsilon=epsilon,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A,
        num_improvements_per_plot=1500,
        pdf_save_path=tempfile.NamedTemporaryFile(delete=False).name
    )
