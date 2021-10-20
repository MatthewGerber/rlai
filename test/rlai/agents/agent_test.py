import pytest
from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents import Human
from rlai.agents.mdp import StochasticMdpAgent
from rlai.policies.tabular import TabularPolicy
from rlai.states.mdp import MdpState


def test_agent_invalid_action():

    random = RandomState()
    agent = StochasticMdpAgent('foo', random, TabularPolicy(None, None), 1.0)

    # test None action
    agent.__act__ = lambda t: None

    with pytest.raises(ValueError, match='Agent returned action of None'):
        agent.act(0)

    # test infeasible action
    action = Action(1, 'foo')
    agent.__act__ = lambda t: action
    state = MdpState(1, [], False)
    agent.sense(state, 0)
    with pytest.raises(ValueError, match=f'Action {action} is not feasible in state {state}'):
        agent.act(0)


def test_human_agent():

    agent = Human()

    a1 = Action(0, 'Foo')
    a2 = Action(1, 'Bar')

    state = MdpState(1, [a1, a2], False)
    agent.sense(state, 0)

    call_num = 0

    def mock_input(
            prompt: str
    ) -> str:

        nonlocal call_num
        if call_num == 0:
            call_num += 1
            return 'asdf'
        else:
            return 'Bar'

    agent.get_input = mock_input  # MagicMock(return_value='Bar')

    assert agent.act(0) == a2

    with pytest.raises(NotImplementedError):
        rng = RandomState(12345)
        Human.init_from_arguments([], rng, None)
