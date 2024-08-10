import numpy as np
import pytest
from numpy.random import RandomState

from rlai.core.environments.gridworld import Gridworld
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.tabular import TabularPolicy
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator


def test_invalid_get_state_i():
    """
    Test.
    """

    policy = TabularPolicy(None, None)

    with pytest.raises(ValueError, match='Attempted to discretize a continuous state without a resolution.'):
        policy.get_state_i(np.array([[1, 2, 3]]))

    with pytest.raises(ValueError, match=f'Unknown state space type:  {type(3)}'):
        # noinspection PyTypeChecker
        policy.get_state_i(3)  # type: ignore


def test_policy_not_equal():
    """
    Test.
    """

    policy_1 = TabularPolicy(None, None)
    policy_2 = TabularPolicy(None, None)

    assert not (policy_1 != policy_2)


# noinspection PyTypeChecker
def test_invalid_epsilon():
    """
    Test.
    """

    with pytest.raises(ValueError, match='epsilon must be >= 0'):
        TabularStateActionValueEstimator(None, -1, None)  # type: ignore


def test_invalid_improve_policy_with_q_pi():
    """
    Test.
    """

    random_state = RandomState(12345)
    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)
    epsilon = 0.0
    mdp_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        TabularStateActionValueEstimator(mdp_environment, epsilon, None)
    )

    assert isinstance(mdp_agent.pi, TabularPolicy)

    with pytest.raises(ValueError, match='Epsilon must be >= 0'):
        mdp_agent.pi.improve_with_q_pi(
            {},
            -1.0
        )
