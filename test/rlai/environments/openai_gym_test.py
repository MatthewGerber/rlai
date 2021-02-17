import os
import pickle

import pytest
from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.openai_gym import Gym, CartpoleFeatureExtractor
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.value_estimation.tabular import TabularStateActionValueEstimator
from test.rlai.utils import tabular_estimator_legacy_eq, tabular_pi_legacy_eq


def test_learn():

    random_state = RandomState(12345)

    gym = Gym(
        random_state=random_state,
        T=None,
        gym_id='CartPole-v1'
    )

    q_S_A = TabularStateActionValueEstimator(gym, 0.05, 0.001)

    mdp_agent = StochasticMdpAgent(
        'agent',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=gym,
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
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gym.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gym.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, fixture_pi) and tabular_estimator_legacy_eq(q_S_A, fixture_q_S_A)


def test_invalid_gym_arguments():

    with pytest.raises(ValueError, match='Continuous-action discretization is only valid for Box action-space environments.'):
        Gym(RandomState(), None, 'CartPole-v0', 0.1)

    with pytest.raises(ValueError, match='render_every_nth_episode must be > 0 if provided.'):
        Gym(RandomState(), None, 'CartPole-v0', render_every_nth_episode=-1)


def test_unimplemented_feature_names():

    cartpole_environment = Gym(RandomState(), None, 'CartPole-v0')
    cartpole_fex = CartpoleFeatureExtractor(cartpole_environment)

    assert cartpole_fex.get_feature_names() is None


def test_video_directory_permisson_error():

    # attempt to use directory off root to raise permission error. should be caught and warned about.
    Gym(RandomState(), None, 'CartPole-v0', render_every_nth_episode=1, video_directory=os.path.abspath(f'{os.sep}test'))
