import logging
from argparse import ArgumentParser
from typing import Optional, Iterator, List, Tuple, Dict, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from numpy.random import RandomState
from patsy.highlevel import dmatrix

from rlai.core import Policy, Action, MdpState, MdpAgent
from rlai.core.environments.mdp import MdpEnvironment
from rlai.docs import rl_text
from rlai.gpi import PolicyImprovementEvent
from rlai.gpi.state_action_value import ValueEstimator, ActionValueEstimator, StateActionValueEstimator
from rlai.gpi.state_action_value.function_approximation.models import (
    StateActionFunctionApproximationModel,
    StateActionFeatureExtractor
)
from rlai.gpi.state_action_value.function_approximation.models.sklearn import SKLearnSGD
from rlai.models.feature_extraction import StationaryFeatureScaler
from rlai.utils import parse_arguments, load_class, log_with_border


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

        self.estimator.add_sample(self.state, self.action, value, weight)
        self.estimator.update_count += 1

    def get_value(
            self
    ) -> float:
        """
        Get current estimated value.

        :return: Value.
        """

        return float(self.estimator.evaluate(self.state, [self.action])[0])

    def __init__(
            self,
            estimator: 'ApproximateStateActionValueEstimator',
            state: MdpState,
            action: Action
    ):
        """
        Initialize the estimator.

        :param estimator: State-action value estimator.
        :param state: State.
        :param action: Action.
        """

        self.estimator = estimator
        self.state = state
        self.action = action


@rl_text(chapter='Value Estimation', page=195)
class ApproximateActionValueEstimator(ActionValueEstimator):
    """
    Approximate action-value estimator.
    """

    def __init__(
            self,
            estimator: 'ApproximateStateActionValueEstimator',
            state: MdpState
    ):
        """
        Initialize the estimator.

        :param estimator: State-action value estimator.
        :param state: State.
        """

        self.estimator = estimator
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
            '--formula',
            type=str,
            help=(
                'Right-hand side of the Patsy-style formula to use when modeling the state-action value relationships. '
                'Ignore to directly use the output of the feature extractor. Note that the use of Patsy formulas will '
                'dramatically slow learning performance, since it is called at each time step.'
            )
        )

        parser.add_argument(
            '--plot-model',
            action='store_true',
            help='Pass this flag to plot the model (e.g., coefficients, diagnostics, etc.).'
        )

        parser.add_argument(
            '--plot-model-per-improvements',
            type=int,
            help=(
                'Number of policy improvements between plots of the state-action value model. Ignore to only plot the '
                'model at the end.'
            )
        )

        parser.add_argument(
            '--plot-model-bins',
            type=int,
            help='Number of bins used when plotting the model. Ignore for no binning.'
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState,
            environment: MdpEnvironment
    ) -> Tuple[StateActionValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
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
            environment=environment,
            model=model,
            feature_extractor=fex,
            **vars(parsed_args)
        )

        return estimator, unparsed_args

    def __init__(
            self,
            environment: MdpEnvironment,
            epsilon: Optional[float],
            model: StateActionFunctionApproximationModel,
            feature_extractor: StateActionFeatureExtractor,
            formula: Optional[str],
            plot_model: bool,
            plot_model_per_improvements: Optional[int],
            plot_model_bins: Optional[int]
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
        `rlai.gpi.state_action_value.function_approximation.models.sklearn.SKLearnSGD`. Online learning
        has implications for the use and coding of categorical variables in the model formula. In particular, the full
        ranges of state and action levels must be specified up front. See
        `test.rlai.gpi.temporal_difference.iteration_test.test_q_learning_iterate_value_q_pi_function_approximation` for
        an example of how this is done. If it is not convenient or possible to specify all state and action levels up
        front, then avoid using categorical variables in the model formula. The variables referenced by the model
        formula must be extracted with identical names by the feature extractor. Although model formulae are convenient
        they are also inefficiently processed in the online fashion described above. Patsy introduces significant
        overhead with each call to the formula parser. A faster alternative is to avoid formula specification (pass
        None here) and return the feature matrix directly from the feature extractor as a numpy.ndarray.
        :param plot_model: Whether to plot the model.
        :param plot_model_per_improvements: Number of policy improvements between plots of the model. Only used
        if `plot_model` is True. Pass None to only plot the model at the end.
        :param plot_model_bins: Number of plotting bins. Only used if `plot_model` is True. Pass None for no binning.
        """

        super().__init__(
            environment=environment,
            epsilon=epsilon
        )

        self.model = model
        self.feature_extractor = feature_extractor
        self.formula = formula
        self.plot_model = plot_model
        self.plot_model_per_improvements = plot_model_per_improvements
        self.plot_model_bins = plot_model_bins

        self.experience_states: List[MdpState] = []
        self.experience_actions: List[Action] = []
        self.experience_values: List[float] = []
        self.weights: Optional[np.ndarray] = None
        self.experience_pending: bool = False
        self.value_scaler = StationaryFeatureScaler()

    def get_initial_policy(
            self
    ) -> 'FunctionApproximationPolicy':
        """
        Get the initial policy defined by the estimator.

        :return: Policy.
        """

        return FunctionApproximationPolicy(self)

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset the estimator for a new run.

        :param state: Initial state.
        """

        super().reset_for_new_run(state)

        self.feature_extractor.reset_for_new_run(state)

    def add_sample(
            self,
            state: MdpState,
            action: Action,
            value: float,
            weight: Optional[float]
    ):
        """
        Add a sample of experience to the estimator. The collection of samples will be used to fit the function
        approximation model when `improve_policy` is called.

        :param state: State.
        :param action: Action.
        :param value: Value.
        :param weight: Weight.
        """

        self.experience_states.append(state)
        self.experience_actions.append(action)
        self.experience_values.append(value)

        if weight is not None:
            if self.weights is None:
                self.weights = np.array([weight])
            else:
                self.weights = np.append(self.weights, [weight], axis=0)

        self.experience_pending = True

    def improve_policy(
            self,
            agent: MdpAgent,
            states: Optional[Iterable[MdpState]],
            event: PolicyImprovementEvent
    ) -> int:
        """
        Improve an agent's policy using the current sample of experience collected through calls to `add_sample`.

        :param agent: Agent whose policy should be improved.
        :param states: States to improve, or None for all states.
        :param event: Event that triggered the improvement.
        :return: Number of states improved.
        """

        super().improve_policy(
            agent=agent,
            states=states,
            event=event
        )

        # if we have pending experience, then fit the model and reset the data.
        if self.experience_pending:

            # extract features and fit the scaler while doing so
            state_action_feature_matrix = self.extract_features(
                self.experience_states,
                self.experience_actions,
                True
            )

            # feature extractors may return a matrix with no columns if extraction was not possible. standardize the
            # outcome values.
            if state_action_feature_matrix.shape[1] > 0:
                self.model.fit(
                    feature_matrix=state_action_feature_matrix,
                    outcomes=self.value_scaler.scale_features(
                        np.array(self.experience_values).reshape(-1, 1),
                        True
                    ).flatten(),
                    weights=self.weights
                )

            self.experience_states.clear()
            self.experience_actions.clear()
            self.experience_values.clear()
            self.weights = None
            self.experience_pending = False

        log_with_border(logging.DEBUG, 'Policy improved')

        return -1 if states is None else len(list(states))

    def evaluate(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> np.ndarray:
        """
        Evaluate the estimator's function approximation model at a state for a list of actions.

        :param state: State.
        :param actions: Actions to evaluate.
        :return: Vector of action values (#actions,).
        """

        log_with_border(logging.DEBUG, f'Evaluating {len(actions)} action(s)')

        # replicate the state for each action, in order to evaluate each state-action pair. don't allow the feature
        # scaler to refit, since it needs to be stationary during evaluation.
        state_action_feature_matrix = self.extract_features([state] * len(actions), actions, False)

        # feature extractors may return a matrix with no columns if extraction was not possible
        if state_action_feature_matrix.shape[1] == 0:  # pragma no cover
            return np.repeat(0.0, len(actions))

        # invert the state-action value back to the original space
        action_values = self.value_scaler.invert_scaled_features(
            self.model.evaluate(state_action_feature_matrix).reshape((-1, 1))
        ).flatten()

        log_with_border(logging.DEBUG, 'Evaluation complete')

        return action_values

    def extract_features(
            self,
            states: List[MdpState],
            actions: List[Action],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is only
        appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the scaler
        should remain fixed, which means this should be False.
        :return: State-feature numpy.ndarray.
        """

        state_action_feature_matrix = self.feature_extractor.extract(states, actions, refit_scaler)

        # if no formula, then the feature extraction result must be a numpy.ndarray to be used directly.
        if self.formula is None:
            if not isinstance(state_action_feature_matrix, np.ndarray):  # pragma no cover
                raise ValueError('Expected feature extractor to return a numpy.ndarray if not a pandas.DataFrame')

        # formulas only work with dataframes
        elif isinstance(state_action_feature_matrix, pd.DataFrame):
            state_action_feature_matrix = dmatrix(self.formula, state_action_feature_matrix)

        # invalid otherwise
        else:
            raise ValueError(
                f'Invalid combination of formula {self.formula} and feature extractor result '
                f'{type(state_action_feature_matrix)}'
            )

        assert isinstance(state_action_feature_matrix, np.ndarray)

        return state_action_feature_matrix

    def plot(
            self,
            final: bool,
            pdf: Optional[PdfPages]
    ) -> Optional[plt.Figure]:
        """
        Plot the estimator. If called from the main thread, then the rendering schedule will be checked and a new plot
        will be generated per the schedule. If called from a background thread, then the data used by the plot will be
        updated but a plot will not be generated or updated. This supports a pattern in which a background thread
        generates new plot data, and a UI thread (e.g., in a Jupyter notebook) periodically calls `update_plot` to
        redraw the plot with the latest data.

        :param final: Whether this is the final time plot will be called.
        :param pdf: PDF for plots, or None for no PDF.
        :return: Matplotlib Figure, if one was generated and not plotting to PDF.
        """

        if self.plot_model:

            render = (
                final or
                (
                    self.plot_model_per_improvements is not None and
                    self.evaluation_policy_improvement_count % self.plot_model_per_improvements == 0
                )
            )

            return self.model.plot(
                feature_extractor=self.feature_extractor,
                policy_improvement_count=self.evaluation_policy_improvement_count,
                num_improvement_bins=self.plot_model_bins,
                render=render,
                pdf=pdf
            )

        return None

    def update_plot(
            self,
            time_step_detail_iteration: Optional[int]
    ):
        """
        Update the plot of the estimator. Can only be called from the main thread.

        :param time_step_detail_iteration: Iteration for which to plot time-step-level detail, or None for no detail.
        Passing -1 will plot detail for the most recently completed iteration.
        """

        assert isinstance(self.model, SKLearnSGD)

        self.model.sklearn_sgd.update_plot(time_step_detail_iteration)

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

        if not isinstance(other, ApproximateStateActionValueEstimator):
            raise ValueError(f'Expected {ApproximateStateActionValueEstimator}')

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


@rl_text(chapter=10, page=244)
class FunctionApproximationPolicy(Policy):
    """
    Policy for use with function approximation methods. This is effectively an interface to the underlying function
    approximation estimator and its reward model, which are accessed by indexing the policy with a state (e.g., a call
    like `agent.pi[state]`), which returns an action-probability dictionary.
    """

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset the policy for a new run.

        :param state: Initial state.
        """

        super().reset_for_new_run(state)

        self.estimator.reset_for_new_run(state)

    def format_state_action_probs(
            self,
            states: List[MdpState]
    ) -> str:
        """
        Get a formatted string containing state-action probabilities for a list of states.

        :param states: States.
        :return: String.
        """

        s = ''
        for state in states:
            s += f'{state}\n'
            for action, prob in self[state].items():
                s += f'\tPr(A={action.name}):  {prob}\n'

        return s

    def format_state_action_values(
            self,
            states: List[MdpState]
    ) -> str:
        """
        Get a formatted string containing state-action values for a list of states.

        :param states: States.
        :return: String.
        """

        s = ''
        for state in states:
            s += f'{state}\n'
            for action, value in zip(state.AA, self.estimator.evaluate(state, state.AA)):
                s += f'\tq(S={state}, A={action.name}):  {value}\n'

        return s

    def __init__(
            self,
            estimator: ApproximateStateActionValueEstimator
    ):
        """
        Initialize the policy.

        :param estimator: State-action value estimator.
        """

        self.estimator = estimator

    def __contains__(
            self,
            state: Optional[MdpState]
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

        if state is None:
            raise ValueError('Attempted to check for None in policy.')

        return True

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state, accounting for the current value of epsilon that is stored in the
        estimator associated with this policy.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        values = self.estimator.evaluate(state, state.AA)
        max_value = max(values)
        num_maximizers = sum(value == max_value for value in values)
        action_prob = {
            action: (
                ((1.0 - self.estimator.epsilon) / num_maximizers) if value == max_value
                else 0.0
            ) + self.estimator.epsilon / len(values)
            for action, value in zip(state.AA, values)
        }

        return action_prob

    def __eq__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current function approximation policy equals another. Two such policies are equal if their
        associated estimators are equal.

        :param other: Other estimator.
        :return: True if equal and False otherwise.
        """

        if not isinstance(other, FunctionApproximationPolicy):
            raise ValueError(f'Expected {FunctionApproximationPolicy}')

        return self.estimator == other.estimator

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current estimator does not equal another.

        :param other: Other estimator.
        :return: True if not equal and False otherwise.
        """

        return not (self == other)
