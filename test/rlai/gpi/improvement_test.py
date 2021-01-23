import pytest
from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.gridworld import Gridworld
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.value_estimation.tabular import TabularStateActionValueEstimator


def test_invalid_improve_policy_with_q_pi():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.0

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    # target agent
    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    with pytest.raises(ValueError, match='Epsilon must be >= 0'):
        improve_policy_with_q_pi(
            mdp_agent,
            {},
            -1
        )
