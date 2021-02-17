import pytest

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.gridworld import Gridworld
from rlai.gpi.temporal_difference.evaluation import evaluate_q_pi, Mode
from numpy.random import RandomState

from rlai.value_estimation.tabular import TabularStateActionValueEstimator


def test_evaluate_q_pi_invalid_n_steps():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, None)

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mdp_environment, epsilon, None)

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    with pytest.raises(ValueError):
        evaluate_q_pi(
            agent=mdp_agent,
            environment=mdp_environment,
            num_episodes=5,
            num_updates_per_improvement=None,
            alpha=0.1,
            mode=Mode.Q_LEARNING,
            n_steps=-1,
            planning_environment=None,
            q_S_A=q_S_A
        )
