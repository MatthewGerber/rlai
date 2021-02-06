import pytest
from numpy.random import RandomState

from rlai.actions import Action
from rlai.environments.gridworld import GridworldFeatureExtractor, Gridworld
from rlai.states.mdp import MdpState


def test_check_state_and_action_lists():

    random = RandomState(12345)
    gw = Gridworld.example_4_1(random, T=None)
    fex = GridworldFeatureExtractor(gw)

    states = [MdpState(i=None, AA=[], terminal=False)]
    actions = [Action(0)]
    fex.check_state_and_action_lists(states, actions)

    with pytest.raises(ValueError, match='Expected '):
        actions.clear()
        fex.check_state_and_action_lists(states, actions)
