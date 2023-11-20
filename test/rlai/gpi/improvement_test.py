import pytest
from numpy.random import RandomState

from rlai.agents.mdp import ActionValueMdpAgent
from rlai.environments.gridworld import Gridworld
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.q_S_A.tabular import TabularStateActionValueEstimator


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

    with pytest.raises(ValueError, match='Epsilon must be >= 0'):
        improve_policy_with_q_pi(
            mdp_agent,
            {},
            -1
        )
