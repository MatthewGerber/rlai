import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import Gridworld
from rlai.gpi.monte_carlo.evaluation import evaluate_v_pi, evaluate_q_pi
from rlai.value_estimation.tabular import TabularStateActionValueEstimator
from test.rlai.utils import tabular_estimator_legacy_eq


def test_evaluate_v_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    v_pi = evaluate_v_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_episodes=1000
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_value.pickle', 'wb') as file:
    #     pickle.dump(v_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert v_pi == fixture


def test_evaluate_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent.initialize_equiprobable_policy(mdp_environment.SS)

    q_S_A = TabularStateActionValueEstimator(mdp_environment)

    evaluated_states, _ = evaluate_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_episodes=1000,
        exploring_starts=True,
        update_upon_every_visit=False,
        q_S_A=q_S_A
    )

    assert len(q_S_A) == len(evaluated_states) + 2  # terminal states aren't evaluated
    assert all(s in q_S_A for s in evaluated_states)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_action_value.pickle', 'wb') as file:
    #     pickle.dump(q_S_A, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_evaluation_of_state_action_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert tabular_estimator_legacy_eq(q_S_A, fixture)
