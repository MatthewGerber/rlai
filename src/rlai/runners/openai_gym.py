from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.openai_gym import Gym
from rlai.gpi.temporal_difference.evaluation import Mode
from rlai.gpi.temporal_difference.iteration import iterate_value_q_pi


def main():

    random_state = RandomState(12345)

    mdp_agent = StochasticMdpAgent(
        'agent',
        random_state,
        0.5
    )

    gym = Gym(
        gym_id='CartPole-v1',
        random_state=random_state,
        continuous_state_discretization_resolution=0.1
    )

    iterate_value_q_pi(
        agent=mdp_agent,
        environment=gym,
        num_improvements=1000,
        num_episodes_per_improvement=50,
        alpha=0.5,
        mode=Mode.Q_LEARNING,
        n_steps=None,
        epsilon=0.05,
        num_improvements_per_plot=20
    )


if __name__ == '__main__':
    main()
