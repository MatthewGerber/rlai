import pytest
from numpy.random import RandomState

from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.core.environments.gridworld import Gridworld
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator, TabularPolicy


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
