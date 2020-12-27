import os
import pickle
import tempfile

from numpy.random import RandomState

from rlai.agents import Human
from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mancala import Mancala, Pit
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.gpi.utils import resume_from_checkpoint
from rlai.policies.tabular import TabularPolicy
from rlai.value_estimation.tabular import TabularStateActionValueEstimator
from test.rlai.utils import tabular_pi_legacy_eq


def test_learn():

    random_state = RandomState(12345)

    mancala: Mancala = Mancala(
        random_state=random_state,
        T=None,
        initial_count=4,
        player_2=StochasticMdpAgent(
            'player 2',
            random_state,
            TabularPolicy(None, None),
            1
        )
    )

    q_S_A = TabularStateActionValueEstimator(mancala, None)

    p1 = StochasticMdpAgent(
        'player 1',
        random_state,
        q_S_A.get_initial_policy(),
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
        make_final_policy_greedy=False,
        q_S_A=q_S_A,
        num_improvements_per_checkpoint=3,
        checkpoint_path=checkpoint_path
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'wb') as file:
    #     pickle.dump(p1.pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(p1.pi, fixture)

    resumed_p1 = resume_from_checkpoint(
        checkpoint_path=checkpoint_path,
        resume_function=iterate_value_q_pi,
        num_improvements=2
    )

    # run same number of improvements without checkpoint...result should be the same.
    random_state = RandomState(12345)

    mancala: Mancala = Mancala(
        random_state=random_state,
        T=None,
        initial_count=4,
        player_2=StochasticMdpAgent(
            'player 2',
            random_state,
            TabularPolicy(None, None),
            1
        )
    )

    q_S_A = TabularStateActionValueEstimator(mancala, None)

    no_checkpoint_p1 = StochasticMdpAgent(
        'player 1',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    iterate_value_q_pi(
        agent=no_checkpoint_p1,
        environment=mancala,
        num_improvements=5,
        num_episodes_per_improvement=100,
        update_upon_every_visit=False,
        epsilon=0.05,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    assert no_checkpoint_p1.pi == resumed_p1.pi


def test_pit():

    pit = Pit(True, 5, True)
    pit.i = 0

    assert str(pit) == '0:  Player 1, 5*'


def test_human_player_mutator():

    random = RandomState()
    mancala = Mancala(random, None, 5, StochasticMdpAgent('foo', random, TabularPolicy(None, []), 1.0))
    Mancala.human_player_mutator(mancala)

    assert isinstance(mancala.player_2, Human)
