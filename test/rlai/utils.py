import os
from typing import Dict

import numpy as np
from xvfbwrapper import Xvfb

from rlai.core import Policy, Action, MdpState
from rlai.gpi.state_action_value.tabular import TabularStateActionValueEstimator, TabularPolicy
from rlai.utils import IncrementalSampleAverager


def tabular_pi_legacy_eq(
        pi: Policy,
        fixture: Dict[MdpState, Dict[Action, float]]
) -> bool:
    """
    Check equality.

    :param pi: Policy.
    :param fixture: Fixture.
    :return: True if equal.
    """

    assert isinstance(pi, TabularPolicy)

    if len(pi) == len(fixture):
        for s in pi:
            if len(pi[s]) == len(fixture[s]):
                for a in pi[s]:
                    if not np.isclose(pi[s][a], fixture[s][a]):
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
    Our older fixtures use a nested dictionary structure (see the type of other above) to store state-action value
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
                    if not np.isclose(estimator.q_S_A[s][a].get_value(), fixture[s][a].get_value()):
                        return False
            else:
                return False
    else:
        return False

    return True


VIRTUAL_DISPLAY = None


def start_virtual_display_if_headless():
    """
    Start a new virtual display if running in headless mode.
    """

    global VIRTUAL_DISPLAY

    if os.getenv('HEADLESS') == 'True' and VIRTUAL_DISPLAY is None:
        VIRTUAL_DISPLAY = Xvfb()
        VIRTUAL_DISPLAY.start()
