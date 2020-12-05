import os
import pickle

from numpy.random import RandomState

from rlai.actions import Action
from rlai.agents.mdp import StochasticMdpAgent
from rlai.environments.mdp import GamblersProblem, PrioritizedSweepingMdpPlanningEnvironment
from rlai.gpi.dynamic_programming.iteration import iterate_value_v_pi
from rlai.planning.environment_models import StochasticEnvironmentModel
from rlai.states.mdp import MdpState


def test_gamblers_problem():

    random_state = RandomState(12345)

    mdp_environment: GamblersProblem = GamblersProblem(
        'gamblers problem',
        random_state=random_state,
        T=None,
        p_h=0.4
    )

    mdp_agent_v_pi_value_iteration = StochasticMdpAgent(
        'test',
        random_state,
        None,
        1
    )

    mdp_agent_v_pi_value_iteration.initialize_equiprobable_policy(mdp_environment.SS)

    v_pi = iterate_value_v_pi(
        mdp_agent_v_pi_value_iteration,
        mdp_environment,
        0.001,
        1,
        True
    )

    # pickle doesn't like to unpickle instances with custom __hash__ functions
    v_pi = {
        s.i: v_pi[s]
        for s in v_pi
    }

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'wb') as file:
    #     pickle.dump(v_pi, file)

    with open(f'{os.path.dirname(__file__)}/fixtures/test_gamblers_problem.pickle', 'rb') as file:
        fixture = pickle.load(file)

    assert v_pi == fixture


def test_prioritized_planning_environment():

    rng = RandomState(12345)

    planning_environment = PrioritizedSweepingMdpPlanningEnvironment(
        'test',
        rng,
        StochasticEnvironmentModel(),
        1,
        0.3,
        10
    )

    planning_environment.add_state_action_priority(MdpState(1, [], False), Action(1), 0.2)
    planning_environment.add_state_action_priority(MdpState(2, [], False), Action(2), 0.1)
    planning_environment.add_state_action_priority(MdpState(3, [], False), Action(3), 0.3)

    s, a = planning_environment.get_state_action_with_highest_priority()
    assert s.i == 2 and a.i == 2
    s, a = planning_environment.get_state_action_with_highest_priority()
    assert s.i == 1 and a.i == 1
    s, a = planning_environment.get_state_action_with_highest_priority()
    assert s is None and a is None
