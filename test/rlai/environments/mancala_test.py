import os
import pickle
import tempfile

from numpy.random import RandomState

from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mancala import Mancala
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi, resume_iterate_value_q_pi_from_checkpoint


def test_learn():

    random_state = RandomState(12345)

    p2 = StochasticMdpAgent(
        'player 2',
        random_state,
        1
    )

    mancala: Mancala = Mancala(
        initial_count=4,
        random_state=random_state,
        player_2=p2
    )

    p1 = StochasticMdpAgent(
        'player 1',
        random_state,
        1
    )

    checkpoint_path = tempfile.NamedTemporaryFile(delete=False).name

    iterate_value_q_pi(
        agent=p1,
        environment=mancala,
        num_improvements=3,
        num_episodes_per_improvement=100,
        update_upon_every_visit=False,
        epsilon=0.05,
        num_improvements_per_checkpoint=3,
        checkpoint_path=checkpoint_path
    )

    # uncomment the following line and run test to update fixture
    with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'wb') as file:
        pickle.dump(p1.pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert p1.pi == fixture

    resumed_p1 = resume_iterate_value_q_pi_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_improvements=2
    )

    # run same number of improvements without checkpoint...result should be the same.
    random_state = RandomState(12345)

    p2 = StochasticMdpAgent(
        'player 2',
        random_state,
        1
    )

    mancala: Mancala = Mancala(
        initial_count=4,
        random_state=random_state,
        player_2=p2
    )

    no_checkpoint_p1 = StochasticMdpAgent(
        'player 1',
        random_state,
        1
    )

    iterate_value_q_pi(
        agent=no_checkpoint_p1,
        environment=mancala,
        num_improvements=5,
        num_episodes_per_improvement=100,
        update_upon_every_visit=False,
        epsilon=0.05
    )

    assert no_checkpoint_p1.pi == resumed_p1.pi



