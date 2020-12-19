from typing import Optional, Iterable

from patsy.highlevel import dmatrices

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.value_estimation import ValueEstimator, ActionValueEstimator, StateActionValueEstimator
from rlai.value_estimation.function_approximation.models import FunctionApproximationModel, FeatureExtractor


@rl_text(chapter='Value Estimation', page=195)
class ApproximateValueEstimator(ValueEstimator):
    """
    Approximate value estimator.
    """

    def update(
            self,
            value: float,
            weight: Optional[float] = None
    ):
        self.estimator.fit(self.state, self.action, value, weight)

    def get_value(
            self
    ) -> float:

        return self.estimator.evaluate(self.state, self.action)

    def __init__(
            self,
            estimator,
            state: MdpState,
            action: Action
    ):
        self.estimator: ApproximateStateActionValueEstimator = estimator
        self.state = state
        self.action = action


@rl_text(chapter='Value Estimation', page=195)
class ApproximateActionValueEstimator(ActionValueEstimator):

    def __init__(
            self,
            estimator,
            state: MdpState
    ):
        self.estimator: ApproximateStateActionValueEstimator = estimator
        self.state = state

    def __getitem__(
            self,
            action: Action
    ) -> ApproximateValueEstimator:

        return ApproximateValueEstimator(self.estimator, self.state, action)


@rl_text(chapter='Value Estimation', page=195)
class ApproximateStateActionValueEstimator(StateActionValueEstimator):

    def initialize(
            self,
            state: MdpState,
            a: Action,
            alpha: Optional[float],
            weighted: bool
    ):
        pass

    def update_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            epsilon: float
    ) -> int:
        pass

    def fit(
            self,
            state: MdpState,
            a: Action,
            value: float,
            weight: float
    ):
        df = self.feature_extractor.extract(state, a)
        df['y'] = value
        X, y = dmatrices(self.formula, df)
        self.model.fit(X, y, weight)

    def evaluate(
            self,
            state: MdpState,
            action: Action
    ) -> float:
        pass

    def __init__(
            self,
            model: FunctionApproximationModel,
            feature_extractor: FeatureExtractor,
            formula: str
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.formula = formula

    def __getitem__(
            self,
            state: MdpState
    ) -> ApproximateActionValueEstimator:

        return ApproximateActionValueEstimator(self, state)
