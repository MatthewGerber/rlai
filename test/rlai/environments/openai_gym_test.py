import os
import pickle

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.openai_gym import Gym
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi
from test.rlai.utils import get_pi_fixture, get_q_S_A_fixture


def test_learn():

    random_state = RandomState(12345)

    mdp_agent = StochasticMdpAgent(
        'agent',
        random_state,
        0.001,
        1
    )

    gym = Gym(
        random_state=random_state,
        T=None,
        gym_id='CartPole-v1'
    )

    q_S_A = iterate_value_q_pi(
        agent=mdp_agent,
        environment=gym,
        num_improvements=10,
        num_episodes_per_improvement=100,
        alpha=0.1,
        mode=Mode.SARSA,
        n_steps=1,
        epsilon=0.05,
        num_planning_improvements_per_direct_improvement=None,
        make_final_policy_greedy=False
    )

    pi = get_pi_fixture(mdp_agent.pi)
    q_S_A = get_q_S_A_fixture(q_S_A)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gym.pickle', 'wb') as file:
    #     pickle.dump((pi, q_S_A), file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gym.pickle', 'rb') as file:
        fixture_pi, fixture_q_S_A = pickle.load(file)

    assert pi == fixture_pi and q_S_A == fixture_q_S_A
