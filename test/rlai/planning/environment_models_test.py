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

    model = StochasticEnvironmentModel(
        random_state=random_state
    )

    actions = [
        Action(i)
        for i in range(5)
    ]

    states = [
        State(i, actions)
        for i in range(100)
    ]

    for t in range(1000):

        state = sample_list_item(states, None, random_state)
        action = sample_list_item(state.AA, None, random_state)

        next_state = sample_list_item(states, None, random_state)
        reward = Reward(None, random_state.random())

        model.update(state, action, next_state, reward)

    state_reward_sequence = []
    curr_state = list(model.state_action_next_state_reward_count.keys())[0]
    for i in range(1000):
        action = sample_list_item(model.state_action_next_state_reward_count[curr_state], None, random_state)
        next_state, reward = model.sample(curr_state, action)
        state_reward_sequence.append((next_state, reward))
        curr_state = next_state

    # uncomment the following line and run test to update fixture
    with open(f'{os.path.dirname(__file__)}/fixtures/test_stochastic_environment_model.pickle', 'wb') as file:
        pickle.dump(state_reward_sequence, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_stochastic_environment_model.pickle', 'rb') as file:
        state_reward_sequence_fixture = pickle.load(file)

    assert state_reward_sequence == state_reward_sequence_fixture
