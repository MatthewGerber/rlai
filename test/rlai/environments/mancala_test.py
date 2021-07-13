import os
import pickle
import tempfile

import pytest
from numpy.random import RandomState

from rlai.agents import Human
from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mancala import Mancala, Pit
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.gpi.utils import resume_from_checkpoint
from rlai.policies.tabular import TabularPolicy
from rlai.utils import sample_list_item
from rlai.q_S_A.tabular import TabularStateActionValueEstimator
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

    q_S_A = TabularStateActionValueEstimator(mancala, 0.05, None)

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
        planning_environment=None,
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

    q_S_A = TabularStateActionValueEstimator(mancala, 0.05, None)

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
        planning_environment=None,
        make_final_policy_greedy=False,
        q_S_A=q_S_A
    )

    assert no_checkpoint_p1.pi == resumed_p1.pi


def test_pit():

    pit = Pit(True, 5, True)
    pit.i = 0

    assert str(pit) == '0:  Player 1, 5*'

    with pytest.raises(ValueError, match='Cannot pick empty pocket.'):
        Pit(True, 0, False).pick()


def test_human_player_mutator():

    random = RandomState()
    mancala = Mancala(random, None, 5, StochasticMdpAgent('foo', random, TabularPolicy(None, []), 1.0))
    Mancala.human_player_mutator(mancala)

    assert isinstance(mancala.player_2, Human)


def test_human_player():

    random_state = RandomState(12345)

    human = Human()

    def mock_input(prompt):
        s = human.most_recent_state
        selected_a = sample_list_item(s.AA, probs=None, random_state=random_state)
        return selected_a.name

    human.get_input = mock_input

    mancala: Mancala = Mancala(
        random_state=random_state,
        T=None,
        initial_count=4,
        player_2=human
    )

    epsilon = 0.05

    q_S_A = TabularStateActionValueEstimator(mancala, epsilon, None)

    p1 = StochasticMdpAgent(
        'player 1',
        random_state,
        q_S_A.get_initial_policy(),
        1
    )

    state = mancala.reset_for_new_run(p1)
    p1.reset_for_new_run(state)
    a = p1.act(0)
    state, reward = mancala.advance(state, 0, a, p1)

    assert mancala.board[7].count == 0 and state.i == 1 and reward.i == 2
