import math
from argparse import ArgumentParser
from typing import Optional, List, Tuple, Iterator, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState
from patsy.highlevel import dmatrix

from rlai.actions import Action
from rlai.agents.mdp import MdpAgent
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies.function_approximation import FunctionApproximationPolicy
from rlai.states.mdp import MdpState
from rlai.utils import load_class, parse_arguments
from rlai.value_estimation import ValueEstimator, ActionValueEstimator, StateActionValueEstimator
from rlai.value_estimation.function_approximation.models import FunctionApproximationModel
from rlai.value_estimation.function_approximation.models.feature_extraction import (
    FeatureExtractor,
    StateActionInteractionFeatureExtractor
)


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
            help='Right-hand side of the Patsy-style formula to use when modeling the state-action value relationships. Ignore to directly use the output of the feature extractor. Note that the use of Patsy formulas will dramatically slow learning performance, since it is called at each time step.'
        )

        parser.add_argument(
            '--plot-model',
            action='store_true',
            help='Whether or not to plot the model (e.g., coefficients).'
        )

        parser.add_argument(
            '--plot-model-per-improvements',
            type=int,
            help='Number of policy improvements between plots of the state-action value model. Ignore to only plot the model at the end.'
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
            environment: MdpEnvironment,
            epsilon: Optional[float]
    ) -> Tuple[StateActionValueEstimator, List[str]]:
        """
        Initialize a state-action value estimator from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :param environment: Environment.
        :param epsilon: Epsilon.
        :return: 2-tuple of a state-action value estimator and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        model_class = load_class(parsed_args.function_approximation_model)
        model, unparsed_args = model_class.init_from_arguments(
            unparsed_args,
            random_state=random_state
        )
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
            states: Optional[Set[MdpState]],
            epsilon: Optional[float]
    ) -> int:
        """
        Improve an agent's policy using the current sample of experience collected through calls to `add_sample`.

        :param agent: Agent whose policy should be improved.
        :param states: States to improve, or None for all states.
        :param epsilon: Epsilon.
        :return: Number of states improved.
        """

        if epsilon is None:
            epsilon = 0.0

        self.epsilon = epsilon

        # if we have pending experience, then fit the model and reset the data.
        if self.experience_pending:

            X = self.get_X(self.experience_states, self.experience_actions, True)
            self.model.fit(X, self.experience_values, self.weights)

            self.experience_states.clear()
            self.experience_actions.clear()
            self.experience_values.clear()
            self.weights = None
            self.experience_pending = False

        if self.policy_improvement_count is None:
            self.policy_improvement_count = 1
        else:
            self.policy_improvement_count += 1

        return -1 if states is None else len(states)

    def evaluate(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> np.ndarray:
        """
        Evaluate the estimator's function approximation model at a state for a list of actions.

        :param state: State.
        :param actions: Actions to evaluate.
        :return: Numpy array of evaluation results (state-action values).
        """

        # replicate the state for each action to evaluate each state-action pair
        X = self.get_X([state] * len(actions), actions, False)

        return self.model.evaluate(X)

    def get_X(
            self,
            states: List[MdpState],
            actions: List[Action],
            for_fitting: bool
    ) -> np.ndarray:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param for_fitting: Whether the extracted features will be used for fitting (True) or prediction (False).
        :return: State-feature numpy.ndarray.
        """

        X = self.feature_extractor.extract(states, actions, for_fitting)

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

    def plot(
            self,
            final: bool
    ):
        """
        Plot the estimator.

        :param final: Whether or not this is the final time plot will be called.
        """

        if self.plot_model:

            model_summary = self.model.get_summary(self.feature_extractor)

            if 'n' in model_summary.columns:
                raise ValueError('Feature extractor returned disallowed column:  n')

            if 'bin' in model_summary.columns:
                raise ValueError('Feature extractor returned disallowed column:  bin')

            model_summary['n'] = self.policy_improvement_count - 1

            if self.plot_df is None:
                self.plot_df = model_summary
            else:
                self.plot_df = self.plot_df.append(model_summary, ignore_index=True)

            if final or (self.plot_model_per_improvements is not None and self.policy_improvement_count % self.plot_model_per_improvements == 0):

                if self.plot_model_bins is None:
                    improvements_per_bin = 1
                    self.plot_df['bin'] = self.plot_df.n
                else:
                    improvements_per_bin = math.ceil(self.policy_improvement_count / self.plot_model_bins)
                    self.plot_df['bin'] = [int(n / improvements_per_bin) for n in self.plot_df.n]

                if isinstance(self.feature_extractor, StateActionInteractionFeatureExtractor):

                    # set up plots
                    feature_names = self.plot_df.feature_name.unique().tolist()
                    n_rows = len(feature_names)
                    action_names = [a.name for a in self.feature_extractor.actions]
                    n_cols = len(action_names)
                    fig, axs = plt.subplots(
                        nrows=n_rows,
                        ncols=n_cols,
                        sharex='all',
                        sharey='row',
                        figsize=(3 * n_cols, 3 * n_rows)
                    )

                    # plot one row per feature, with actions as the columns
                    for i, feature_name in enumerate(feature_names):
                        feature_df = self.plot_df[self.plot_df.feature_name == feature_name]
                        boxplot_axs = axs[i, :]
                        feature_df.boxplot(column=action_names, by='bin', ax=boxplot_axs)
                        boxplot_axs[0].set_ylabel(f'w({feature_name})')

                    # reset labels and titles
                    for i, row in enumerate(axs):
                        for ax in row:

                            if i < axs.shape[0] - 1:
                                ax.set_xlabel('')
                            else:
                                ax.set_xlabel('Iteration' if self.plot_model_bins is None else f'Bin of {improvements_per_bin} iterations')

                            if i > 0:
                                ax.set_title('')

                    fig.suptitle('Model coefficients over iterations')
                    plt.tight_layout()
                    plt.show()

                else:
                    raise ValueError(f'Unknown feature extractor type:  {type(self.feature_extractor)}')

    def __init__(
            self,
            environment: MdpEnvironment,
            epsilon: Optional[float],
            model: FunctionApproximationModel,
            feature_extractor: FeatureExtractor,
            formula: Optional[str],
            plot_model: Optional[bool],
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
        :param plot_model: Whether or not to plot the model.
        :param plot_model_per_improvements: Number of policy improvements between between plots.
        :param plot_model_bins: Number of plotting bins, or None for no binning.
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
        self.weights: np.ndarray = None
        self.experience_pending: bool = False
        self.policy_improvement_count: Optional[int] = None
        self.plot_df: Optional[pd.DataFrame] = None

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
