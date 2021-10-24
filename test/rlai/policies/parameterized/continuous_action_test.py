import numpy as np
from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.openai_gym import Gym
from rlai.v_S.function_approximation.models.feature_extraction import ContinuousFeatureExtractor
from rlai.policies.parameterized.continuous_action import ContinuousActionBetaDistributionPolicy


def test_rescale():

    random_state = RandomState(12345)

    gym = Gym(
        random_state=random_state,
        T=None,
        gym_id='LunarLanderContinuous-v2'
    )

    fex = ContinuousFeatureExtractor()

    policy = ContinuousActionBetaDistributionPolicy(gym, fex, False)

    agent = StochasticMdpAgent('test', random_state, policy, 0.9)
    state = gym.reset_for_new_run(agent)
    policy.set_action(state)

    assert np.allclose(policy.rescale(np.array([0.0, 0.5])), np.array([-1.0, 0.0]))
    assert np.allclose(policy.rescale(np.array([0.5, 1.0])), np.array([0.0, 1.0]))

    assert np.allclose(policy.invert_rescale(policy.rescale(np.array([0.0, 0.5]))), np.array([0.0, 0.5]))
    assert np.allclose(policy.invert_rescale(policy.rescale(np.array([0.5, 1.0]))), np.array([0.5, 1.0]))
