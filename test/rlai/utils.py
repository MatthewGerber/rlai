from typing import Dict

from rlai.actions import Action
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager


def get_pi_fixture(
        pi: Dict[MdpState, Dict[Action, float]]
) -> Dict[int, Dict[Action, float]]:
    """
    pickle doesn't like to unpickle instances with custom __hash__ functions. Get something more pickle-able.

    :param pi: Policy.
    :return: Pickle-able policy.
    """

    return {
        s.i: {
            a: pi[s][a]
            for a in pi[s]
        }
        for s in pi
    }


def get_q_S_A_fixture(
        q_S_A: Dict[MdpState, Dict[Action, IncrementalSampleAverager]]
) -> Dict[int, Dict[Action, float]]:
    """
    pickle doesn't like to unpickle instances with custom __hash__ functions. Get something more pickle-able.

    :param q_S_A: Q-values.
    :return: Pickle-able q-values.
    """

    return {
        s.i: {
            a: q_S_A[s][a].get_value()
            for a in q_S_A[s]
        }
        for s in q_S_A
    }
