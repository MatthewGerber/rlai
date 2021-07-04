import logging
from argparse import ArgumentParser
from typing import Optional, List, Tuple

import numpy as np
from numpy.random import RandomState

from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.utils import load_class, parse_arguments, log_with_border
from rlai.v_S import ValueEstimator, StateValueEstimator
from rlai.v_S.function_approximation.models import FunctionApproximationModel
from rlai.v_S.function_approximation.models.feature_extraction import FeatureExtractor


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

        return self.estimator.evaluate(self.state)[0]

    def __init__(
            self,
            estimator,
            state: MdpState
    ):
        """
        Initialize the estimator.

        :param estimator: State-action value estimator.
        :param state: State.
        """

        self.estimator: ApproximateStateValueEstimator = estimator
        self.state = state


@rl_text(chapter='Value Estimation', page=195)
class ApproximateStateValueEstimator(StateValueEstimator):
    """
    Approximate state value estimator.
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

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[StateValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # load model
        model_class = load_class(parsed_args.function_approximation_model)
        model, unparsed_args = model_class.init_from_arguments(
            unparsed_args,
            random_state=random_state
        )
        del parsed_args.function_approximation_model

        # load feature extractor
        feature_extractor_class = load_class(parsed_args.feature_extractor)
        fex, unparsed_args = feature_extractor_class.init_from_arguments(unparsed_args)
        del parsed_args.feature_extractor

        # initialize estimator
        estimator = ApproximateStateValueEstimator(
            model=model,
            feature_extractor=fex,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

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

        super().improve_policy()

        # if we have pending experience, then fit the model and reset the data.
        if self.experience_pending:

            X = self.get_X(self.experience_states, True)

            # feature extractors may return a matrix with no columns if extraction was not possible
            if X.shape[1] > 0:
                self.model.fit(X, self.experience_values, self.weights)

            self.experience_states.clear()
            self.experience_values.clear()
            self.weights = None
            self.experience_pending = False

        log_with_border(logging.DEBUG, 'Estimator improved')

    def evaluate(
            self,
            state: MdpState
    ) -> float:
        """
        Evaluate the estimator's function approximation model at a state.

        :param state: State.
        :return: Estimate.
        """

        log_with_border(logging.DEBUG, f'Evaluating state')

        # get feature vector
        X = self.get_X([state], False)

        # feature extractors may return a matrix with no columns if extraction was not possible
        if X.shape[1] == 0:  # pragma no cover
            return 0.0

        value = self.model.evaluate(X)

        log_with_border(logging.DEBUG, 'Evaluation complete')

        return value

    def get_X(
            self,
            states: List[MdpState],
            for_fitting: bool
    ) -> np.ndarray:
        """
        Extract features for states.

        :param states: States.
        :param for_fitting: Whether the extracted features will be used for fitting (True) or prediction (False).
        :return: State-feature numpy.ndarray.
        """

        return self.feature_extractor.extract(states, for_fitting)

    def __init__(
            self,
            model: FunctionApproximationModel,
            feature_extractor: FeatureExtractor
    ):
        """
        Initialize the estimator.

        :param model: Model.
        :param feature_extractor: Feature extractor.
        """

        super().__init__()

        self.model = model
        self.feature_extractor = feature_extractor

        self.experience_states: List[MdpState] = []
        self.experience_values: List[float] = []
        self.weights: np.ndarray = None
        self.experience_pending: bool = False

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
        :return: True if defined and False otherise.
        """

        return True

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator equals another.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """

        other: ApproximateStateValueEstimator

        return self.model == other.model

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
