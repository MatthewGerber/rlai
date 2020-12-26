from argparse import Namespace, ArgumentParser
from typing import Optional, List, Tuple, Iterator, Set

import numpy as np
import pandas as pd
from patsy.highlevel import dmatrix

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies.function_approximation import FunctionApproximationPolicy
from rlai.states.mdp import MdpState
from rlai.utils import load_class
from rlai.value_estimation import ValueEstimator, ActionValueEstimator, StateActionValueEstimator
from rlai.value_estimation.function_approximation.statistical_learning import FunctionApproximationModel
from rlai.value_estimation.function_approximation.statistical_learning.feature_extraction import FeatureExtractor


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

        self.estimator.fit(self.state, self.action, value, weight)

    def get_value(
            self
    ) -> float:
        """
        Get current estimated value.

        :return: Value.
        """

        return self.estimator.evaluate(self.state, [self.action])[0]

    def __init__(
            self,
            estimator,
            state: MdpState,
            action: Action
    ):
        """
        Initialize the estimator.

        :param estimator: State-action value estimator.
        :param state: State.
        :param action: Action.
        """

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
        """
        Initialize the estimator.

        :param estimator: State-action value estimator.
        :param state: State.
        """

        self.estimator: ApproximateStateActionValueEstimator = estimator
        self.state = state

    def __getitem__(
            self,
            action: Action
    ) -> ApproximateValueEstimator:
        """
        Get value estimator for an action.

        :param action: Action.
        :return: Value estimator.
        """

        return ApproximateValueEstimator(self.estimator, self.state, action)

    def __len__(
            self
    ) -> int:
        """
        Get number of actions defined by the estimator.

        :return: Number of actions.
        """

        return len(self.state.AA)

    def __iter__(
            self
    ) -> Iterator[Action]:
        """
        Get iterator over actions.

        :return: Iterator.
        """

        return iter(self.state.AA)

    def __contains__(
            self,
            action: Action
    ) -> bool:
        """
        Check whether action is defined.

        :param action: Action.
        :return: True if defined and False otherwise.
        """

        return action in self.state.AA_set


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
            '--formula',
            type=str,
            help='Right-hand side of the Patsy-style formula to use when modeling the state-action value relationships. Ignore to directly use the output of the feature extractor. Note that the use of Patsy formulas will dramatically slow learning performance, since it is called at each time step.'
        )

        return parser.parse_known_args(args)

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment,
            epsilon: Optional[float]
    ) -> Tuple[StateActionValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :param epsilon: Epsilon.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        model_class = load_class(parsed_args.function_approximation_model)
        model, unparsed_args = model_class.init_from_arguments(unparsed_args)
        del parsed_args.function_approximation_model

        feature_extractor_class = load_class(parsed_args.feature_extractor)
        fex, unparsed_args = feature_extractor_class.init_from_arguments(unparsed_args, environment)
        del parsed_args.feature_extractor

        estimator = ApproximateStateActionValueEstimator(
            environment=environment,
            epsilon=epsilon,
            model=model,
            feature_extractor=fex,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

    def get_initial_policy(
            self
    ) -> FunctionApproximationPolicy:
        """
        Get the initial policy defined by the estimator.

        :return: Policy.
        """

        return FunctionApproximationPolicy(self)

    def update_policy(
            self,
            agent: MdpAgent,
            states: Optional[Set[MdpState]],
            epsilon: float
    ) -> int:
        """
        Update an agent's policy using the current state-action value estimates.

        :param agent: Agent whose policy should be updated.
        :param states: States to update, or None for all states.
        :param epsilon: Epsilon.
        :return: Number of states updated.
        """

        # nothing to do here, as we've already updated the function approximation model through calls to fit. just
        # update the value of epsilon (e.g., in case it's being made greedy) and return.

        self.epsilon = epsilon

        return -1 if states is None else len(states)

    def fit(
            self,
            state: MdpState,
            action: Action,
            value: float,
            weight: Optional[float]
    ):
        """
        Update the estimator's function approximation model.

        :param state: State.
        :param action: Action.
        :param value: Value.
        :param weight: Weight.
        """

        X = self.get_X(state, [action])
        y = np.array([value])

        self.model.fit(X, y, weight)

    def evaluate(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> np.ndarray:
        """
        Evaluate the estimator's function approximation model at a state for a list of actions.

        :param state: State.
        :param actions: Actions to evaluate.
        :return: Numpy array of evaluation results (values).
        """

        X = self.get_X(state, actions)

        return self.model.evaluate(X)

    def get_X(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> np.ndarray:
        """
        Get feature matrix for a state and list of actions.

        :param state: State.
        :param actions: Actions.
        :return: Feature matrix (#obs, #features).
        """

        X = self.feature_extractor.extract(state, actions)

        # if no formula, then use result directly.
        if self.formula is None:
            if isinstance(X, pd.DataFrame):
                X = X.to_numpy()
            elif not isinstance(X, np.ndarray):
                raise ValueError('Expected feature extractor to return a numpy.ndarray if not a pandas.DataFrame')

        # use formula with dataframe only
        elif isinstance(X, pd.DataFrame):
            X = dmatrix(self.formula, X)

        # invalid otherwise
        else:
            raise ValueError(f'Invalid combination of formula {self.formula} and feature extractor result {type(X)}')

        return X

    def __init__(
            self,
            environment: MdpEnvironment,
            epsilon: Optional[float],
            model: FunctionApproximationModel,
            feature_extractor: FeatureExtractor,
            formula: Optional[str]
    ):
        """
        Initialize the estimator.

        :param environment: Environment.
        :param epsilon: Epsilon.
        :param model: Model.
        :param feature_extractor: Feature extractor.
        :param formula: Model formula. This is only the right-hand side of the model. If you want to implement a model
        like "r ~ x + y + z" (i.e., to model reward as a linear function of features x, y, and z), then you should pass
        "x + y + z" for this argument. See the Patsy documentation for full details of the formula language. Statistical
        learning models used in reinforcement learning generally need to operate "online", learning the reward function
        incrementally at each step. An example of such a model would be
        `rlai.value_estimation.function_approximation.statistical_learning.sklearn.SKLearnSGD`. Online learning has
        implications for the use and coding of categorical variables in the model formula. In particular, the full
        ranges of state and action levels must be specified up front. See
        `test.rlai.gpi.temporal_difference.iteration_test.test_q_learning_iterate_value_q_pi_function_approximation` for
        an example of how this is done. If it is not convenient or possible to specify all state and action levels up
        front, then avoid using categorical variables in the model formula. The variables referenced by the model
        formula must be extracted with identical names by the feature extractor. Althrough model formulae are convenient
        they are also inefficiently processed in the online fashion described above. Patsy introduces significant
        overhead with each call to the formula parser. A faster alternative is to avoid formula specification (pass
        None here) and return the feature matrix directly from the feature extractor as a numpy.ndarray.
        """

        if epsilon is None:
            epsilon = 0.0

        self.environment = environment
        self.epsilon = epsilon
        self.model = model
        self.feature_extractor = feature_extractor
        self.formula = formula

    def __getitem__(
            self,
            state: MdpState
    ) -> ApproximateActionValueEstimator:
        """
        Get the action-value estimator for a state.

        :param state: State.
        :return: Action-value estimator.
        """

        return ApproximateActionValueEstimator(self, state)

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

        other: ApproximateStateActionValueEstimator

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
