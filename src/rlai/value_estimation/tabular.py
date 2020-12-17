from typing import Dict, Optional, Iterable

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.gpi.improvement import improve_policy_with_q_pi
from rlai.states.mdp import MdpState
from rlai.utils import IncrementalSampleAverager
from rlai.value_estimation import StateActionValueEstimator, ActionValueEstimator, ValueEstimator


class TabularValueEstimator(ValueEstimator):

    def update(
            self,
            value: float,
            weight: Optional[float] = None
    ):
        self.averager.update(
            value=value,
            weight=weight
        )

    def get_value(
            self
    ) -> float:

        return self.averager.get_value()

    def __init__(
            self,
            alpha: float,
            weighted: bool
    ):
        self.averager = IncrementalSampleAverager(
            alpha=alpha,
            weighted=weighted
        )


class TabularActionValueEstimator(ActionValueEstimator):

    def __init__(
            self
    ):
        self.q_A: Dict[Action, TabularValueEstimator] = {}

    def __contains__(
            self,
            action: Action
    ) -> bool:

        return action in self.q_A

    def __getitem__(
            self,
            action: Action
    ) -> TabularValueEstimator:

        return self.q_A[action]

    def __setitem__(
            self,
            action: Action,
            value_estimator: TabularValueEstimator
    ):
        self.q_A[action] = value_estimator

    def __len__(
            self
    ) -> int:

        return len(self.q_A)


class TabularStateActionValueEstimator(StateActionValueEstimator):

    def initialize(
            self,
            state: MdpState,
            a: Action,
            alpha: Optional[float],
            weighted: bool
    ):
        if state not in self:
            self[state] = TabularActionValueEstimator()

        if a not in self[state]:
            self[state][a] = TabularValueEstimator(alpha=alpha, weighted=weighted)

    def update_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            epsilon: float
    ) -> int:
        q_pi = {
            s: {
                a: self[s][a].get_value()
                for a in self[s].q_A
            }
            for s in agent.pi
            if states is None or s in states
        }

        num_states_updated = improve_policy_with_q_pi(
            agent=agent,
            q_pi=q_pi,
            epsilon=epsilon
        )

        return num_states_updated

    def __init__(
            self
    ):
        self.q_S_A: Dict[MdpState, TabularActionValueEstimator] = {}

    def __contains__(
            self,
            state: MdpState
    ) -> bool:

        return state in self.q_S_A
    
    def __getitem__(
            self,
            state: MdpState
    ) -> TabularActionValueEstimator:

        return self.q_S_A[state]

    def __setitem__(
            self,
            state: MdpState,
            action_value_estimator: TabularActionValueEstimator
    ):
        self.q_S_A[state] = action_value_estimator

    def __len__(
            self
    ) -> int:

        return len(self.q_S_A)
