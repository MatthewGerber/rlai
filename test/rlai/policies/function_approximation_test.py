import pytest
from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.gridworld import Gridworld, GridworldFeatureExtractor
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from rlai.q_S_A.function_approximation.estimators import ApproximateStateActionValueEstimator
from rlai.q_S_A.function_approximation.models.sklearn import SKLearnSGD
from rlai.states.mdp import MdpState


def test_policy_overrides():

    random_state = RandomState(12345)

    mdp_environment: Gridworld = Gridworld.example_4_1(random_state, 20)

    epsilon = 0.05

    q_S_A = ApproximateStateActionValueEstimator(
        mdp_environment,
        epsilon,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        GridworldFeatureExtractor(mdp_environment),
        None,
        False,
        None,
        None
    )

    mdp_agent = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=mdp_environment,
        num_improvements=10,
        num_episodes_per_improvement=20,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A
    )

    random_state = RandomState(12345)

    mdp_environment_2: Gridworld = Gridworld.example_4_1(random_state, 20)

    q_S_A_2 = ApproximateStateActionValueEstimator(
        mdp_environment_2,
        epsilon,
        SKLearnSGD(random_state=random_state, scale_eta0_for_y=False),
        GridworldFeatureExtractor(mdp_environment_2),
        None,
        False,
        None,
        None
    )

    mdp_agent_2 = StochasticMdpAgent(
        'test',
        random_state,
        q_S_A_2.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=mdp_agent_2,
        environment=mdp_environment_2,
        num_improvements=10,
        num_episodes_per_improvement=20,
        num_updates_per_improvement=None,
        alpha=None,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        planning_environment=None,
        make_final_policy_greedy=True,
        q_S_A=q_S_A_2
    )
    
    assert isinstance(mdp_agent_2.most_recent_state, MdpState) and mdp_agent_2.most_recent_state in mdp_agent_2.pi

    with pytest.raises(ValueError, match='Attempted to check for None in policy.'):
        # noinspection PyTypeChecker
        if None in mdp_agent_2.pi:
            pass

    assert mdp_agent.pi == mdp_agent_2.pi
    assert not (mdp_agent.pi != mdp_agent_2.pi)
