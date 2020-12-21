from argparse import Namespace, ArgumentParser
from typing import Optional, Iterable, List, Tuple, Iterator

from patsy.highlevel import dmatrices

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies.function_approximation import FunctionApproximationPolicy
from rlai.states.mdp import MdpState
from rlai.utils import load_class
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
    """
    Approximate action-value estimator.
    """

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

    def __len__(
            self
    ) -> int:

        return len(self.state.AA)

    def __iter__(
            self
    ) -> Iterator[Action]:

        return self.state.AA

    def __contains__(self, action: Action) -> bool:

        return self.state.is_feasible(action)


@rl_text(chapter='Value Estimation', page=195)
class ApproximateStateActionValueEstimator(StateActionValueEstimator):
    """
    Approximate state-action value estimator.
    """

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parser = ArgumentParser(allow_abbrev=False)

        # future arguments for this base class can be added here...

        return parser.parse_known_args(args)

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[StateActionValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        model_class = load_class(parsed_args.function_approximation_model)
        model, unparsed_args = model_class.init_from_arguments(unparsed_args)
        del parsed_args.function_approximation_model

        feature_extractor_class = load_class(parsed_args.feature_extractor)
        fex, unparsed_args = feature_extractor_class.init_from_arguments(unparsed_args)
        del parsed_args.feature_extractor

        estimator = ApproximateStateActionValueEstimator(
            environment=environment,
            model=model,
            feature_extractor=fex,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

    def get_initial_policy(
            self
    ) -> FunctionApproximationPolicy:

        return FunctionApproximationPolicy()

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
            weight: Optional[float]
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
            environment: MdpEnvironment,
            model: FunctionApproximationModel,
            feature_extractor: FeatureExtractor,
            formula: str
    ):
        self.environment = environment
        self.model = model
        self.feature_extractor = feature_extractor
        self.formula = formula

    def __getitem__(
            self,
            state: MdpState
    ) -> ApproximateActionValueEstimator:

        return ApproximateActionValueEstimator(self, state)

    def __len__(
            self
    ) -> int:

        return 1

    def __contains__(
            self,
            state: MdpState
    ) -> bool:

        return True

    def __eq__(
            self,
            other
    ) -> bool:

        other: ApproximateStateActionValueEstimator

        return self.model == other.model

    def __ne__(
            self,
            other
    ) -> bool:

        return not (self == other)
