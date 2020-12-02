import os
import pickle

from numpy.random import RandomState

from rlai.actions import Action
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.rewards import Reward
from rlai.states import State
from rlai.utils import sample_list_item


def test_stochastic_environment_model():

    random_state = RandomState(12345)

    model = StochasticEnvironmentModel(None)

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

    # test state-action prioritization
    model.add_state_action_priority(State(1, []), Action(1), 0.2)
    model.add_state_action_priority(State(2, []), Action(2), 0.1)
    model.add_state_action_priority(State(3, []), Action(3), 0.3)
    s, a = model.get_state_action_with_highest_priority()
    assert s.i == 2 and a.i == 2
    s, a = model.get_state_action_with_highest_priority()
    assert s.i == 1 and a.i == 1
    s, a = model.get_state_action_with_highest_priority()
    assert s.i == 3 and a.i == 3
    s, a = model.get_state_action_with_highest_priority()
    assert s is None and a is None
