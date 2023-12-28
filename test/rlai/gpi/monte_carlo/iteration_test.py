import os
import pickle
import tempfile
import time
from threading import Thread

import pytest
from numpy.random import RandomState

from rlai.core.environments.gridworld import Gridworld, GridworldFeatureExtractor
from rlai.core.environments.mdp import TrajectorySamplingMdpPlanningEnvironment, StochasticEnvironmentModel
from rlai.gpi import PolicyImprovementEvent
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.function_approximation import (
    ApproximateStateActionValueEstimator,
    FunctionApproximationPolicy
)
from rlai.gpi.state_action_value.function_approximation.models.sklearn import SKLearnSGD
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator
from rlai.gpi.utils import update_policy_iteration_plot, plot_policy_iteration
from rlai.models.sklearn import SKLearnSGD as BaseSKLearnSGD
from rlai.utils import RunThreadManager
from test.rlai.utils import tabular_estimator_legacy_eq, tabular_pi_legacy_eq


def test_iterate_value_q_pi():
    """
    Test.
    """

    random_state = RandomState(12345)
    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.1, None)
    mdp_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        q_S_A
    )
    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=3000,
        num_episodes_per_improvement=1,
        update_upon_every_visit=False,
        planning_environment=None,
        make_final_policy_greedy=False
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_off_policy_monte_carlo():
    """
    Test.
    """

    # target agent
    random_state = RandomState(12345)
    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.0, None)
    mdp_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        q_S_A
    )

    # episode generation (behavior) policy
    off_policy_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, 0.0, None)
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        update_upon_every_visit=True,
        planning_environment=None,
        make_final_policy_greedy=False,
        off_policy_agent=off_policy_agent
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_off_policy_iteration_of_value_q_pi.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_monte_carlo_off_policy_iteration_of_value_q_pi.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, pi_fixture) and tabular_estimator_legacy_eq(q_S_A, q_S_A_fixture)


def test_off_policy_monte_carlo_with_function_approximation():
    """
    Test.
    """

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        0.05,
        SKLearnSGD(BaseSKLearnSGD(random_state=random_state)),
        GridworldFeatureExtractor(mdp_environment),
        None,
        False,
        None,
        None
    )

    # target agent
    mdp_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        q_S_A
    )

    # episode generation (behavior) policy
    off_policy_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, None, None)
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        update_upon_every_visit=True,
        planning_environment=None,
        make_final_policy_greedy=False,
        off_policy_agent=off_policy_agent
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_off_policy_monte_carlo_with_function_approximationo.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_off_policy_monte_carlo_with_function_approximationo.pickle', 'rb') as file:
        pi_fixture, q_S_A_fixture = pickle.load(file)

    assert mdp_agent.pi == pi_fixture and q_S_A == q_S_A_fixture
    assert isinstance(mdp_agent.pi, FunctionApproximationPolicy)
    assert str(mdp_agent.pi.estimator[mdp_environment.SS[5]][mdp_environment.SS[5].AA[1]]).startswith('-3.6144')

    # make greedy
    q_S_A.epsilon = 0.0
    assert q_S_A.improve_policy(mdp_agent, None, PolicyImprovementEvent.MAKING_POLICY_GREEDY) == -1
    assert mdp_agent.pi.estimator.epsilon == 0.0


def test_invalid_iterate_value_q_pi():
    """
    Test.
    """

    # target agent
    random_state = RandomState(12345)
    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
    mdp_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, 0.0, None)
    )

    # episode generation (behavior) policy
    off_policy_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, 0.0, None)
    )

    with pytest.raises(ValueError, match='Planning environments are not currently supported for Monte Carlo iteration.'):
        iterate_value_q_pi(
            agent=mdp_agent,
            environment=mdp_environment,
            num_improvements=100,
            num_episodes_per_improvement=1,
            update_upon_every_visit=True,
            planning_environment=TrajectorySamplingMdpPlanningEnvironment('foo', random_state, StochasticEnvironmentModel(), 100, None),
            make_final_policy_greedy=False,
            off_policy_agent=off_policy_agent
        )

    # test warning...no off-policy agent with epsilon=0.0
    mdp_agent.q_S_A.epsilon = 0.0
    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=100,
        num_episodes_per_improvement=1,
        update_upon_every_visit=True,
        planning_environment=None,
        make_final_policy_greedy=False,
        off_policy_agent=None
    )


def test_iterate_value_q_pi_with_pdf():
    """
    Test.
    """

    random_state = RandomState(12345)
    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
    q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.1, None)
    mdp_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        q_S_A
    )
    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=3000,
        num_episodes_per_improvement=1,
        update_upon_every_visit=False,
        planning_environment=None,
        make_final_policy_greedy=False,
        num_improvements_per_plot=1500,
        pdf_save_path=tempfile.NamedTemporaryFile(delete=False).name
    )

    with pytest.raises(ValueError, match='Epsilon must be >= 0'):
        q_S_A.epsilon = -1.0
        q_S_A.improve_policy(mdp_agent, states=None, event=PolicyImprovementEvent.MAKING_POLICY_GREEDY)

    q_S_A.epsilon = 0.0
    assert q_S_A.improve_policy(mdp_agent, None, PolicyImprovementEvent.MAKING_POLICY_GREEDY) == 14


def test_iterate_value_q_pi_multi_threaded():
    """
    Test.
    """

    thread_manager = RunThreadManager(True)

    def train_thread_target():

        random_state = RandomState(12345)
        mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
        q_S_A = TabularStateActionValueEstimator(mdp_environment, 0.1, None)
        mdp_agent = ActionValueMdpAgent(
            'test',
            random_state,
            1,
            q_S_A
        )
        iterate_value_q_pi(
            agent=mdp_agent,
            environment=mdp_environment,
            num_improvements=1000000,
            num_episodes_per_improvement=10,
            update_upon_every_visit=False,
            planning_environment=None,
            make_final_policy_greedy=False,
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
