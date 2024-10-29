import numpy as np
from numpy.random import RandomState

from rlai.core.environments.gymnasium import Gym, ContinuousLunarLanderFeatureExtractor
from rlai.policy_gradient import ParameterizedMdpAgent
from rlai.policy_gradient.policies.continuous_action import ContinuousActionBetaDistributionPolicy


def test_rescale():
    """
    Test.
    """

    random_state = RandomState(12345)

    gym = Gym(
        random_state=random_state,
        T=None,
        gym_id='LunarLanderContinuous-v3'
    )

    fex = ContinuousLunarLanderFeatureExtractor(True)
    policy = ContinuousActionBetaDistributionPolicy(gym, fex, False)
    # noinspection PyTypeChecker
    agent = ParameterizedMdpAgent('test', random_state, policy, 0.9, None)
    state = gym.reset_for_new_run(agent)
    policy.set_action(state)

    assert np.allclose(policy.rescale(np.array([0.0, 0.5])), np.array([-1.0, 0.0]))
    assert np.allclose(policy.rescale(np.array([0.5, 1.0])), np.array([0.0, 1.0]))

    assert np.allclose(policy.invert_rescale(policy.rescale(np.array([0.0, 0.5]))), np.array([0.0, 0.5]))
    assert np.allclose(policy.invert_rescale(policy.rescale(np.array([0.5, 1.0]))), np.array([0.5, 1.0]))
