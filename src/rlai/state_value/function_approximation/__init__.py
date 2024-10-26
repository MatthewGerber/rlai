from argparse import ArgumentParser
from typing import List, Tuple, Optional

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState

from rlai.core import MdpState, Environment
from rlai.docs import rl_text
from rlai.models.feature_extraction import StationaryFeatureScaler
from rlai.state_value import StateValueEstimator, ValueEstimator
from rlai.state_value.function_approximation.models import StateFunctionApproximationModel
from rlai.state_value.function_approximation.models.feature_extraction import StateFeatureExtractor
from rlai.utils import parse_arguments, load_class


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
        """
        Update the value estimate.

        :param value: New value.
        :param weight: Weight.
        """

        self.estimator.add_sample(self.state, value, weight)
        self.estimator.update_count += 1

    def get_value(
            self
    ) -> float:
        """
        Get current estimated value.

        :return: Value.
        """

        return self.estimator.evaluate(self.state)

    def __init__(
            self,
            estimator: 'ApproximateStateValueEstimator',
            state: MdpState
    ):
        """
        Initialize the estimator.

        :param estimator: State-action value estimator.
        :param state: State.
        """

        self.estimator = estimator
        self.state = state


@rl_text(chapter='Value Estimation', page=195)
class ApproximateStateValueEstimator(StateValueEstimator):
    """
    Approximate state-value estimator.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        parser = ArgumentParser(
            prog=f'{cls.__module__}.{cls.__name__}',
            parents=[super().get_argument_parser()],
            allow_abbrev=False,
            add_help=False
        )

        parser.add_argument(
            '--function-approximation-model',
            type=str,
            help='Fully-qualified type name of function approximation model.'
        )

        parser.add_argument(
            '--feature-extractor',
            type=str,
            help='Fully-qualified type name of feature extractor.'
        )

        parser.add_argument(
            '--scale-outcomes',
            action='store_true',
            help='Whether to scale (standardize) outcomes before fitting the function approximation model.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: Environment
    ) -> Tuple[StateValueEstimator, List[str]]:
        """
        Initialize a state-value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a state-value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # load feature extractor
        feature_extractor_class = load_class(parsed_args.feature_extractor)
        fex, unparsed_args = feature_extractor_class.init_from_arguments(
            args=unparsed_args,
            environment=environment
        )
        del parsed_args.feature_extractor

        # load model
        model_class = load_class(parsed_args.function_approximation_model)
        model, unparsed_args = model_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            fit_intercept=not fex.extracts_intercept()
        )
        del parsed_args.function_approximation_model

        # initialize estimator
        estimator = cls(
            model=model,
            feature_extractor=fex,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

    def __init__(
            self,
            model: StateFunctionApproximationModel,
            feature_extractor: StateFeatureExtractor,
            scale_outcomes: bool
    ):
        """
        Initialize the estimator.

        :param model: Model.
        :param feature_extractor: Feature extractor.
        :param scale_outcomes: Whether to scale state-value outcomes before fitting the estimator model.
        """

        super().__init__()

        self.model = model
        self.feature_extractor = feature_extractor
        self.scale_outcomes = scale_outcomes

        self.experience_states: List[MdpState] = []
        self.experience_values: List[float] = []
        self.weights: Optional[np.ndarray] = None
        self.experience_pending: bool = False
        self.value_scaler = StationaryFeatureScaler()

    def add_sample(
            self,
            state: MdpState,
            value: float,
            weight: Optional[float]
    ):
        """
        Add a sample of experience to the estimator. The collection of samples will be used to fit the function
        approximation model when `improve` is called.

        :param state: State.
        :param value: Value.
        :param weight: Weight.
        """

        self.experience_states.append(state)
        self.experience_values.append(value)

        if weight is not None:
            if self.weights is None:
                self.weights = np.array([weight])
            else:
                self.weights = np.append(self.weights, [weight], axis=0)

        self.experience_pending = True

    def improve(
            self
    ):
        """
        Improve an agent's policy using the current sample of experience collected through calls to `add_sample`.

        :return: Number of states improved.
        """

        # if we have pending experience, then fit the model and reset the data.
        if self.experience_pending:

            state_feature_matrix = self.extract_features(self.experience_states, True)

            outcomes = np.array(self.experience_values)
            if self.scale_outcomes:
                outcomes = self.value_scaler.scale_features(outcomes.reshape(-1, 1), True).flatten()

            # feature extractors may return a matrix with no columns if extraction was not possible
            if state_feature_matrix.shape[1] > 0:
                self.model.fit(
                    feature_matrix=state_feature_matrix,
                    outcomes=outcomes,
                    weights=self.weights
                )

            self.experience_states.clear()
            self.experience_values.clear()
            self.weights = None
            self.experience_pending = False

    def evaluate(
            self,
            state: MdpState
    ) -> float:
        """
        Evaluate the estimator's function approximation model at a state.

        :param state: State.
        :return: Estimate.
        """

        # extract feature matrix
        state_feature_matrix = self.extract_features([state], False)

        # feature extractors may return a matrix with no columns if extraction was not possible
        if state_feature_matrix.shape[1] == 0:  # pragma no cover
            return 0.0

        state_values = self.model.evaluate(state_feature_matrix)

        # invert the state value back to the original space if we're scaling
        if self.scale_outcomes:
            state_values = self.value_scaler.invert_scaled_features(state_values.reshape((-1, 1))).flatten()

        assert len(state_values) == 1

        return float(state_values[0])

    def extract_features(
            self,
            states: List[MdpState],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract features for states.

        :param states: States.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature matrix (#states, #features).
        """

        return self.feature_extractor.extract(states, refit_scaler)

    def plot(
            self,
            pdf: Optional[PdfPages]
    ):
        """
        Plot the current estimator.

        :param pdf: PDF to plot to, or None to show directly.
        """

        self.model.plot(True, pdf)

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset for new run.
        """

        self.feature_extractor.reset_for_new_run(state)

    def __getitem__(
            self,
            state: MdpState
    ) -> ApproximateValueEstimator:
        """
        Get the value estimator for a state.

        :param state: State.
        :return: Value estimator.
        """

        return ApproximateValueEstimator(self, state)

    def __len__(
            self
    ) -> int:
        """
        Get number of states defined by the estimator.

        :return: Number of states.
        """

        # a bit of a hack, as we don't actually track the number of states.
        return 1

    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether a state is defined by the estimator.

        :param state: State.
        :return: True if defined and False otherwise.
        """

        return True

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, ApproximateStateValueEstimator):
            raise ValueError(f'Expected {ApproximateStateValueEstimator}')

        return self.model == other.model

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
