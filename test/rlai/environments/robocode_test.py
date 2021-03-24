import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.robocode import RobocodeEnvironment, RobocodeFeatureExtractor
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.value_estimation.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.value_estimation.function_approximation.models.sklearn import SKLearnSGD


def test_learn():

    random_state = RandomState(12345)

    robocode = RobocodeEnvironment(
        random_state=random_state,
        T=None,
        port=54321
    )

    q_S_A = ApproximateStateActionValueEstimator(
        robocode,
        0.15,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        RobocodeFeatureExtractor(robocode),
        None,
        False,
        None,
        None
    )

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        0.9
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=robocode,
        num_improvements=10,
        num_episodes_per_improvement=1,
        num_updates_per_improvement=1,
        alpha=None,
        mode=Mode.SARSA,
        n_steps=100,
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    # uncomment the following line and run test to update fixture
    with open(f'{os.path.dirname(__file__)}/fixtures/test_robocode.pickle', 'wb') as file:
        pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_robocode.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert mdp_agent.pi == fixture_pi and q_S_A == fixture_q_S_A
