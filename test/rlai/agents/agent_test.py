import pytest

from rlai.actions import Action
from rlai.agents.mdp import StochasticMdpAgent
from numpy.random import RandomState

from rlai.policies.tabular import TabularPolicy
from rlai.states.mdp import MdpState


def test_agent_invalid_action():

    random = RandomState()
    agent = StochasticMdpAgent('foo', random, TabularPolicy(None, None), 1.0)

    # test None action
    agent.__act__ = lambda t: None

    with pytest.raises(ValueError, match='Agent returned action of None'):
        agent.act(0)

    # test infeasiable action
    action = Action(1, 'foo')
    agent.__act__ = lambda t: action
    state = MdpState(1, [], False)
    agent.sense(state, 0)
    with pytest.raises(ValueError, match=f'Action {action} is not feasible in state {state}'):
        agent.act(0)
