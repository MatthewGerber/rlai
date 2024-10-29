import os
import pickle

import pytest
from numpy.random import RandomState

from rlai.core.environments.gymnasium import Gym, CartpoleFeatureExtractor
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from test.rlai.utils import tabular_estimator_legacy_eq, tabular_pi_legacy_eq


def test_learn():
    """
    Test.
    """

    random_state = RandomState(12345)
    gym = Gym(
        random_state=random_state,
        T=None,
        gym_id='CartPole-v1'
    )
    q_S_A = TabularStateActionValueEstimator(gym, 0.05, 0.001)
    mdp_agent = ActionValueMdpAgent(
        'agent',
        random_state,
        1,
        q_S_A
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
        make_final_policy_greedy=False
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gym.pickle', 'wb') as file:
    #     pickle.dump((mdp_agent.pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gym.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert tabular_pi_legacy_eq(mdp_agent.pi, fixture_pi) and tabular_estimator_legacy_eq(q_S_A, fixture_q_S_A)


def test_invalid_gym_arguments():
    """
    Test.
    """

    with pytest.raises(ValueError, match='Continuous-action discretization is only valid for Box action-space environments.'):
        Gym(RandomState(), None, 'CartPole-v1', 0.1)

    with pytest.raises(ValueError, match='render_every_nth_episode must be > 0 if provided.'):
        Gym(RandomState(), None, 'CartPole-v1', render_every_nth_episode=-1)


def test_unimplemented_feature_names():
    """
    Test.
    """

    cartpole_environment = Gym(RandomState(), None, 'CartPole-v1')
    cartpole_fex = CartpoleFeatureExtractor(cartpole_environment, True)

    assert cartpole_fex.get_action_feature_names() is None
