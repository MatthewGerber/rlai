import os
import pickle
import shlex
import tempfile
import time
from threading import Thread, Event
from typing import Optional

import numpy as np
import pytest
from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.gridworld import Gridworld, GridworldFeatureExtractor
from rlai.environments.mdp import TrajectorySamplingMdpPlanningEnvironment
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.gpi.utils import update_policy_iteration_plot, plot_policy_iteration
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.runners.trainer import run
from rlai.utils import RunThreadManager
from rlai.q_S_A.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.q_S_A.function_approximation.models.feature_extraction import (
    StateActionIdentityFeatureExtractor
)
from rlai.q_S_A.function_approximation.models.sklearn import SKLearnSGD
from rlai.q_S_A.tabular import TabularStateActionValueEstimator
from test.rlai.utils import tabular_estimator_legacy_eq, tabular_pi_legacy_eq


def test_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.05, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_sarsa_iterate_value_q_pi_make_greedy():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.05, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_make_greedy.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_make_greedy.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_sarsa_iterate_value_q_pi_with_trajectory_planning():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.05, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    planning_environment = TrajectorySamplingMdpPlanningEnvironment(
        'test planning',
        random_state,
        StochasticEnvironmentModel(),
        10,
        None
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        planning_environment=planning_environment,
        make_final_policy_greedy=True,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_planning.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_iteration_of_value_q_pi_planning.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_q_learning_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.05, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=1,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_q_learning_iterate_value_q_pi_function_approximation_with_formula():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        0.05,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        StateActionIdentityFeatureExtractor(mdp_environment),
        f'C(s, levels={[s.i for s in mdp_environment.SS]}):C(a, levels={[a.i for a in mdp_environment.SS[0].AA]})',
        False,
        None,
        None
    )

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=5,
        num_episodes_per_improvement=5,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert np.allclose(mdp_agent.pi.estimator.model.model.coef_, pi_fixture.estimator.model.model.coef_)


def test_q_learning_iterate_value_q_pi_function_approximation_no_formula():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        0.05,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        GridworldFeatureExtractor(mdp_environment),
        None,
        False,
        None,
        None
    )

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=20,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation_no_formula.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_q_learning_iterate_value_q_pi_function_approximation_no_formula.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert np.allclose(mdp_agent.pi.estimator.model.model.coef_, pi_fixture.estimator.model.model.coef_)
    assert mdp_agent.pi.format_state_action_probs(mdp_environment.SS) == pi_fixture.format_state_action_probs(mdp_environment.SS)
    assert mdp_agent.pi.format_state_action_values(mdp_environment.SS) == pi_fixture.format_state_action_values(mdp_environment.SS)


def test_q_learning_iterate_value_q_pi_function_approximation_invalid_formula():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        0.05,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        GridworldFeatureExtractor(mdp_environment),
        f'C(s, levels={[s.i for s in mdp_environment.SS]}):C(a, levels={[a.i for a in mdp_environment.SS[0].AA]})',
        False,
        None,
        None
    )

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    with pytest.raises(ValueError, match='Invalid combination of formula'):
        iterate_value_q_pi(
            agent=mdp_agent,
            environment=mdp_environment,
            num_improvements=5,
            num_episodes_per_improvement=5,
            num_updates_per_improvement=None,
            alpha=None,
            mode=Mode.Q_LEARNING,
            n_steps=None,
            planning_environment=None,
            make_final_policy_greedy=False,
            q_S_A=q_S_A
        )


def test_expected_sarsa_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.05, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.EXPECTED_SARSA,
        n_steps=1,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_expected_sarsa_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_n_step_q_learning_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.05, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=3,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_td_n_step_q_learning_iteration_of_value_q_pi.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, fixture_pi) and tabular_estimator_legacy_eq(q_S_A, fixture_q_S_A)


def test_invalid_epsilon_iterate_value_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.0, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    with pytest.raises(ValueError, match='epsilon must be strictly > 0 for TD-learning'):
        iterate_value_q_pi(
            agent=mdp_agent,
            environment=mdp_environment,
            num_improvements=10,
            num_episodes_per_improvement=100,
            num_updates_per_improvement=None,
            alpha=0.1,
            mode=Mode.Q_LEARNING,
            n_steps=3,
            planning_environment=None,
            make_final_policy_greedy=False,
            q_S_A=q_S_A
        )


def test_iterate_value_q_pi_with_pdf():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.05, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=100,
        num_updates_per_improvement=None,
        alpha=0.1,
        mode=Mode.Q_LEARNING,
        n_steps=1,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A,
        num_improvements_per_plot=5,
        pdf_save_path=tempfile.NamedTemporaryFile(delete=False).name
    )


def test_iterate_value_q_pi_multi_threaded():

    thread_manager = RunThreadManager(True)

    def train_thread_target():

        random_state = RandomState(12345)

        mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

        q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.1, None)

        mdp_agent = StochasticMdpAgent(
            'test',
            random_state,
            q_S_A.get_initial_policy(),
            1
        )

        iterate_value_q_pi(
            agent=mdp_agent,
            environment=mdp_environment,
            num_improvements=1000000,
            num_episodes_per_improvement=10,
            num_updates_per_improvement=None,
            alpha=0.1,
            mode=Mode.SARSA,
            n_steps=None,
            planning_environment=None,
            make_final_policy_greedy=False,
            q_S_A=q_S_A,
            thread_manager=thread_manager,
            num_improvements_per_plot=10
        )

    # premature update should have no effect
    assert update_policy_iteration_plot() is None

    # initialize plot from main thread
    plot_policy_iteration(
        iteration_average_reward=[],
        iteration_total_states=[],
        iteration_num_states_improved=[],
        elapsed_seconds_average_rewards={},
        pdf=None
    )

    # run training thread
    run_thread = Thread(target=train_thread_target)
    run_thread.start()
    time.sleep(1)

    # update plot asynchronously
    update_policy_iteration_plot()
    time.sleep(1)

    # should be allowed to update plot from non-main thread
    def bad_update():
        with pytest.raises(ValueError, match='Can only update plot on main thread.'):
            update_policy_iteration_plot()

    bad_thread = Thread(target=bad_update)
    bad_thread.start()
    bad_thread.join()

    thread_manager.abort = True
    run_thread.join()


def test_iterate_value_q_pi_func_approx_multi_threaded():

    thread_manager = RunThreadManager(True)

    train_args_wait_event = Event()

    q_S_A: Optional[ApproximateStateActionValueEstimator] = None

    def train_args_callback(
            train_args
    ):
        nonlocal q_S_A
        q_S_A = train_args['q_S_A']
        train_args_wait_event.set()

    cmd = '--random-seed 12345 --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.gridworld.Gridworld --id example_4_1 --T 25 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 10 --num-episodes-per-improvement 10 --epsilon 0.05 --q-S-A rlai.q_S_A.function_approximation.estimators.ApproximateStateActionValueEstimator --function-approximation-model rlai.q_S_A.function_approximation.models.sklearn.SKLearnSGD --plot-model --feature-extractor rlai.environments.gridworld.GridworldFeatureExtractor --make-final-policy-greedy True'
    args = shlex.split(cmd)

    def train_thread_target():
        run(
            args=args,
            thread_manager=thread_manager,
            train_function_args_callback=train_args_callback
        )

    train_thread = Thread(target=train_thread_target)
    train_thread.start()

    train_args_wait_event.wait()

    # premature update should do nothing
    assert q_S_A.update_plot(-1) is None

    time.sleep(1)
    assert q_S_A.plot(True, None) is not None

    # should be allowed to update plot from non-main thread
    def bad_update():
        with pytest.raises(ValueError, match='Can only update plot on main thread.'):
            q_S_A.update_plot(-1)

    bad_thread = Thread(target=bad_update)
    bad_thread.start()
    bad_thread.join()

    q_S_A.update_plot(-1)


def test_q_learning_iterate_value_q_pi_function_approximation_policy_ne():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    epsilon = 0.05

    q_S_A_1 = ApproximateStateActionValueEstimator(
        mdp_environment,
        epsilon,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        GridworldFeatureExtractor(mdp_environment),
        None,
        False,
        None,
        None
    )

    mdp_agent_1 = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A_1.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent_1,
        environment=mdp_environment,
        num_improvements=5,
        num_episodes_per_improvement=10,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A_1
    )

    q_S_A_2 = ApproximateStateActionValueEstimator(
        mdp_environment,
        epsilon,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        GridworldFeatureExtractor(mdp_environment),
        None,
        False,
        None,
        None
    )

    mdp_agent_2 = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A_2.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent_2,
        environment=mdp_environment,
        num_improvements=5,
        num_episodes_per_improvement=5,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A_2
    )

    assert mdp_agent_1.pi.estimator != mdp_agent_2.pi.estimator
    assert mdp_agent_1.pi.estimator.model != mdp_agent_2.pi.estimator.model


def test_q_learning_iterate_value_q_pi_tabular_policy_ne():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    epsilon = 0.05

    q_S_A_1 = TabularStateActionValueEstimator(
        mdp_environment,
        epsilon,
        None
    )

    mdp_agent_1 = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A_1.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent_1,
        environment=mdp_environment,
        num_improvements=5,
        num_episodes_per_improvement=10,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A_1
    )

    q_S_A_2 = TabularStateActionValueEstimator(
        mdp_environment,
        epsilon,
        None
    )

    mdp_agent_2 = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A_2.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent_2,
        environment=mdp_environment,
        num_improvements=5,
        num_episodes_per_improvement=5,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A_2
    )

    test_state = mdp_environment.SS[5]
    test_action = test_state.AA[0]

    assert q_S_A_1 != q_S_A_2
    assert q_S_A_1[test_state] != q_S_A_2[test_state]
    assert q_S_A_1[test_state][test_action] != q_S_A_2[test_state][test_action]
