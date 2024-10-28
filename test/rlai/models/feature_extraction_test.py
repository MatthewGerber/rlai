import numpy as np
import pytest
from numpy.random import RandomState

from rlai.core import MdpState, Action
from rlai.core.environments.gridworld import Gridworld, GridworldFeatureExtractor
from rlai.models.feature_extraction import OneHotCategory, OneHotCategoricalFeatureInteracter


def test_one_hot_category():
    """
    Test.
    """

    booleans = [True, False]
    ohc_1 = OneHotCategory(*booleans)
    assert str(ohc_1) == '_'.join(str(arg) for arg in booleans)

    ohc_2 = OneHotCategory(*booleans)
    assert ohc_1 == ohc_2


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
    interacter = OneHotCategoricalFeatureInteracter(cats, False)
    with pytest.raises(ValueError, match='Expected '):
        interacter.interact(
            feature_matrix=np.array([
                [1, 2, 3],
                [4, 5, 6]
            ]),
            categorical_values=[1],
            refit_scaler=False
        )
