import os
import pickle

from numpy.random import RandomState
import numpy as np
from rl.agents.mdp import EquiprobableRandom
from rl.dynamic_programming.evaluation import iterative_policy_evaluation_of_state_value, \
    iterative_policy_evaluation_of_action_value
from rl.environments.mdp import Gridworld
import os
import pickle

import numpy as np
from numpy.random import RandomState

from rl.agents.mdp import EquiprobableRandom
from rl.dynamic_programming.evaluation import iterative_policy_evaluation_of_state_value, \
    iterative_policy_evaluation_of_action_value
from rl.environments.mdp import Gridworld


def test_iterative_policy_evaluation_of_state_value():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    mdp_agent = EquiprobableRandom(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    state_value = iterative_policy_evaluation_of_state_value(
        mdp_agent,
        mdp_environment,
        0.001,
        True
    )

    state_value_not_in_place = iterative_policy_evaluation_of_state_value(
        mdp_agent,
        mdp_environment,
        0.001,
        False
    )

    assert list(state_value.keys()) == list(state_value_not_in_place.keys())

    assert np.allclose(list(state_value.values()), list(state_value_not_in_place.values()), atol=0.01)

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    state_value = {
        s.i: state_value[s]
        for s in state_value
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_state_value.pickle', 'wb') as file:
    #     pickle.dump(state_value, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_state_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert state_value == fixture


def test_iterative_policy_evaluation_of_action_value():

    mdp_environment: Gridworld = Gridworld.example_4_1()

    random_state = RandomState(12345)

    mdp_agent = EquiprobableRandom(
        mdp_environment.AA,
        'test',
        random_state,
        mdp_environment.SS,
        1
    )

    action_value = iterative_policy_evaluation_of_action_value(
        mdp_agent,
        mdp_environment,
        0.001,
        True
    )

    action_value_not_in_place = iterative_policy_evaluation_of_action_value(
        mdp_agent,
        mdp_environment,
        0.001,
        False
    )

    assert list(action_value.keys()) == list(action_value_not_in_place.keys())

    for s in action_value:
        for a in action_value[s]:
            assert np.allclose(action_value[s][a], action_value_not_in_place[s][a], atol=0.01)

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    action_value = {
        s.i: {
            a.i: action_value[s][a]
            for a in action_value[s]
        }
        for s in action_value
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_action_value.pickle', 'wb') as file:
    #     pickle.dump(action_value, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_iterative_policy_evaluation_of_action_value.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert action_value == fixture
