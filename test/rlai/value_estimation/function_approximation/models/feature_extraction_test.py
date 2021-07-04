import os
import pickle

import pytest
from numpy.random import RandomState

from rlai.actions import Action
from rlai.environments.gridworld import GridworldFeatureExtractor, Gridworld
from rlai.states.mdp import MdpState
from rlai.q_S_A.function_approximation.models.feature_extraction import OneHotCategoricalFeatureInteracter
from rlai.models.feature_extraction import NonstationaryFeatureScaler
import numpy as np
import numpy.random


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


def test_bad_interact():

    cats = [1, 2]
    interacter = OneHotCategoricalFeatureInteracter(cats)
    with pytest.raises(ValueError, match='Expected '):
        interacter.interact(np.array([
            [1, 2, 3],
            [4, 5, 6]
        ]), [1])


def test_nonstationary_feature_scaler():

    numpy.random.seed(12345)

    scaler = NonstationaryFeatureScaler(100, 10, 0.9)

    for i in range(20):
        X = numpy.random.rand(10, 5)
        scaler.scale_features(X, for_fitting=True)

    X = numpy.random.rand(10, 5)
    X_scaled = scaler.scale_features(X, for_fitting=False)

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_nonstationary_feature_scaler.pickle', 'wb') as file:
    #     pickle.dump(X_scaled, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_nonstationary_feature_scaler.pickle', 'rb') as file:
        X_scaled_fixture = pickle.load(file)

    assert np.allclose(X_scaled, X_scaled_fixture)
