import os
from typing import Dict

from xvfbwrapper import Xvfb

from rlai.actions import Action
from rlai.policies import Policy
from rlai.policies.tabular import TabularPolicy
from rlai.q_S_A.tabular import TabularStateActionValueEstimator
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager


def tabular_pi_legacy_eq(
        pi: Policy,
        fixture: Dict[MdpState, Dict[Action, float]]
) -> bool:

    pi: TabularPolicy

    if len(pi) == len(fixture):
        for s in pi:
            if len(pi[s]) == len(fixture[s]):
                for a in pi[s]:
                    if pi[s][a] != fixture[s][a]:
                        return False
            else:
                return False
    else:
        return False

    return True


def tabular_estimator_legacy_eq(
        estimator: TabularStateActionValueEstimator,
        fixture: Dict[MdpState, Dict[Action, IncrementalSampleAverager]]
) -> bool:
    """
    Our older fixtures use a nested dictionary structure (see the type of "other above) to store state-action value
    estimates. The newer approach uses a class-based structure to support function approximation. This function bridges
    the two for the purposes of test assertions.

    :param estimator: Estimator.
    :param fixture: Fixture.
    :return: True if equal.
    """

    if len(estimator.q_S_A) == len(fixture):
        for s in estimator:
            if len(estimator.q_S_A[s]) == len(fixture[s]):
                for a in estimator.q_S_A[s]:
                    if estimator.q_S_A[s][a].get_value() != fixture[s][a].get_value():
                        return False
            else:
                return False
    else:
        return False

    return True


VIRTUAL_DISPLAY_INITIALIZED = False


def init_virtual_display():
    """
    Initialize a new virtual display if running in headless mode.
    """

    global VIRTUAL_DISPLAY_INITIALIZED

    headless = os.getenv('HEADLESS') == 'True'

    virtual_display = None

    if headless and not VIRTUAL_DISPLAY_INITIALIZED:
        virtual_display = Xvfb(width=1280, height=740, colordepth=16)
        virtual_display.start()
        VIRTUAL_DISPLAY_INITIALIZED = True

    return virtual_display
