import os
import pickle
import tempfile

import pytest
from numpy.random import RandomState

from rlai.core import Human
from rlai.core.environments.mancala import Mancala, Pit
from rlai.gpi.monte_carlo.iteration import iterate_value_q_pi
from rlai.gpi.state_action_value import ActionValueMdpAgent
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator
from rlai.gpi.utils import resume_from_checkpoint
from rlai.utils import sample_list_item
from test.rlai.utils import tabular_pi_legacy_eq


def test_learn():
    """
    Test.
    """

    random_state = RandomState(12345)

    mancala: Mancala = Mancala(
        random_state=random_state,
        T=None,
        initial_count=4,
        player_2=None
    )

    p1 = ActionValueMdpAgent(
        'player 1',
        random_state,
        1,
        TabularStateActionValueEstimator(mancala, 0.05, None)
    )

    checkpoint_path = iterate_value_q_pi(
        agent=p1,
        environment=mancala,
        num_improvements=3,
        num_episodes_per_improvement=100,
        update_upon_every_visit=False,
        planning_environment=None,
        make_final_policy_greedy=False,
        num_improvements_per_checkpoint=3,
        checkpoint_path=tempfile.NamedTemporaryFile(delete=False).name
    )

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'wb') as file:
    #     pickle.dump(p1.pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_mancala.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert tabular_pi_legacy_eq(p1.pi, fixture)

    _, resumed_p1 = resume_from_checkpoint(
        checkpoint_path=checkpoint_path,
        resume_function=iterate_value_q_pi,
        num_improvements=5
    )

    # run same number of improvements without checkpoint...result should be the same.
    random_state = RandomState(12345)
    mancala = Mancala(
        random_state=random_state,
        T=None,
        initial_count=4,
        player_2=None
    )
    no_checkpoint_p1 = ActionValueMdpAgent(
        'player 1',
        random_state,
        1,
        TabularStateActionValueEstimator(mancala, 0.05, None)
    )

    iterate_value_q_pi(
        agent=no_checkpoint_p1,
        environment=mancala,
        num_improvements=5,
        num_episodes_per_improvement=100,
        update_upon_every_visit=False,
        planning_environment=None,
        make_final_policy_greedy=False
    )

    assert no_checkpoint_p1.pi == resumed_p1.pi


def test_pit():
    """
    Test.
    """

    pit = Pit(True, 5, True)
    pit.i = 0

    assert str(pit) == '0:  Player 1, 5*'

    with pytest.raises(ValueError, match='Cannot pick empty pocket.'):
        Pit(True, 0, False).pick()


def test_human_player_mutator():
    """
    Test.
    """

    random = RandomState()
    mancala = Mancala(random, None, 5, None)
    Mancala.human_player_mutator(mancala)

    assert isinstance(mancala.player_2, Human)


def test_human_player():
    """
    Test.
    """

    random_state = RandomState(12345)

    human = Human()

    def mock_input(
            prompt: str
    ) -> str:
        s = human.most_recent_state
        selected_a = sample_list_item(s.AA, probs=None, random_state=random_state)
        return selected_a.name

    human.get_input = mock_input  # type: ignore[method-assign]

    mancala: Mancala = Mancala(
        random_state=random_state,
        T=None,
        initial_count=4,
        player_2=human
    )

    epsilon = 0.05

    p1 = ActionValueMdpAgent(
        'player 1',
        random_state,
        1,
        TabularStateActionValueEstimator(mancala, epsilon, None)
    )

    state = mancala.reset_for_new_run(p1)
    p1.reset_for_new_run(state)
    a = p1.act(0)
    state, reward = mancala.advance(state, 0, a, p1)

    assert mancala.board[7].count == 0 and state.i == 1 and reward.i == 2
