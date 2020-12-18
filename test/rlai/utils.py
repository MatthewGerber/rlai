from typing import Dict

from rlai.actions import Action
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager
from rlai.value_estimation.tabular import TabularStateActionValueEstimator


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