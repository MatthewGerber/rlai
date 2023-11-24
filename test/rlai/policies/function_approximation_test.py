import pytest
from numpy.random import RandomState

from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.core.environments.gridworld import Gridworld, GridworldFeatureExtractor
from rlai.core import MdpState
from rlai.gpi.state_action_value.function_approximation import ApproximateStateActionValueEstimator
from rlai.gpi.state_action_value.function_approximation.models.sklearn import SKLearnSGD
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi


def test_policy_overrides():
    """
    Test.
    """

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

    mdp_agent = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        q_S_A
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

    mdp_agent_2 = ActionValueMdpAgent(
        'test',
        random_state,
        1,
        q_S_A_2
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
        make_final_policy_greedy=True
    )

    assert isinstance(mdp_agent_2.most_recent_state, MdpState) and mdp_agent_2.most_recent_state in mdp_agent_2.pi

    with pytest.raises(ValueError, match='Attempted to check for None in policy.'):
        # noinspection PyTypeChecker
        if None in mdp_agent_2.pi:  # pragma no cover
            pass

    assert mdp_agent.pi == mdp_agent_2.pi
    assert not (mdp_agent.pi != mdp_agent_2.pi)
