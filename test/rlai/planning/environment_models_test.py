import os
import pickle

from numpy.random import RandomState

from rlai.actions import Action
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.rewards import Reward
from rlai.states import State
from rlai.utils import sample_list_item


def test_stochastic_environment_model():
    """
    Test.
    """

    random_state = RandomState(12345)

    model = StochasticEnvironmentModel()

    actions = [
        Action(i)
        for i in range(5)
    ]

    states = [
        State(i, actions)
        for i in range(5)
    ]

    for t in range(1000):
        state = sample_list_item(states, None, random_state)
        action = sample_list_item(state.AA, None, random_state)
        next_state = sample_list_item(states, None, random_state)
        reward = Reward(None, random_state.randint(10))
        model.update(state, action, next_state, reward)

    environment_sequence = []
    for i in range(1000):
        state = model.sample_state(random_state)
        action = model.sample_action(state, random_state)
        next_state, reward = model.sample_next_state_and_reward(state, action, random_state)
        environment_sequence.append((next_state, reward))

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_stochastic_environment_model.pickle', 'wb') as file:
    #     pickle.dump(environment_sequence, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_stochastic_environment_model.pickle', 'rb') as file:
        environment_sequence_fixture = pickle.load(file)

    assert environment_sequence == environment_sequence_fixture
