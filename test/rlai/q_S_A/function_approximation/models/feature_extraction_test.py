import numpy as np
import pytest
from numpy.random import RandomState

from rlai.core import Action, MdpState
from rlai.core.environments.gridworld import GridworldFeatureExtractor, Gridworld
from rlai.models.feature_extraction import OneHotCategoricalFeatureInteracter


def test_check_state_and_action_lists():
    """
    Test.
    """

    random = RandomState(12345)
    gw = Gridworld.example_4_1(random, T=None)
    fex = GridworldFeatureExtractor(gw)

    states = [MdpState(i=None, AA=[], terminal=False, truncated=False)]
    actions = [Action(0)]
    fex.check_state_and_action_lists(states, actions)

    with pytest.raises(ValueError, match='Expected '):
        actions.clear()
        fex.check_state_and_action_lists(states, actions)


def test_bad_interact():
    """
    Test.
    """

    cats = [1, 2]
    interacter = OneHotCategoricalFeatureInteracter(cats)
    with pytest.raises(ValueError, match='Expected '):
        interacter.interact(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]), [1])
