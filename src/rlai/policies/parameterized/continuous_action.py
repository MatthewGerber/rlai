import logging
import warnings
from abc import ABC
from argparse import ArgumentParser
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from jax import numpy as jnp, jit, grad, vmap
from jax.scipy import stats as jstats
from numpy.random import RandomState
from scipy import stats
from tabulate import tabulate

from rlai.actions import Action, ContinuousMultiDimensionalAction
from rlai.environments.mdp import ContinuousMdpEnvironment
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.policies.parameterized import ParameterizedPolicy
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments, load_class, is_positive_definite, ScatterPlot
from rlai.v_S.function_approximation.models.feature_extraction import StateFeatureExtractor


@rl_text(chapter=13, page=335)
class ContinuousActionPolicy(ParameterizedPolicy, ABC):
    """
    Parameterized policy that produces continuous, multi-dimensional actions.
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
            '--policy-feature-extractor',
            type=str,
            help='Fully-qualified type name of feature extractor to use within policy.'
        )

        parser.add_argument(
            '--plot-policy',
            action='store_true',
            help='Pass this flag to plot policy values (e.g., actions taken and their parameters).'
        )

        return parser

    def set_action(
            self,
            state: MdpState
    ):
        """
        Set the single, continuous, multi-dimensional action for this policy based on a state. This function can be
        called repeatedly; however, it will only have an effect upon the first call. It assumes that the state contains
        a single action and will raise an exception if it does not.

        :param state: State.
        """

        if self.action is None:
            if len(state.AA) == 1 and isinstance(state.AA[0], ContinuousMultiDimensionalAction):
                self.action = state.AA[0]
            else:  # pragma no cover
                raise ValueError('Expected state to contain a single action of type ContinuousMultiDimensionalAction.')

    def update_action_scatter_plot(
            self,
            action: ContinuousMultiDimensionalAction
    ):
        """
        Update the action scatter plot.

        :param action: Action.
        """

        if self.action_scatter_plot is not None:
            self.action_scatter_plot.update(action.value)

    def reset_action_scatter_plot_y_range(
            self
    ):
        """
        Reset the y-range in the action scatter plot.
        """

        if self.action_scatter_plot is not None:
            self.action_scatter_plot.reset_y_range()

    def close(
            self
    ):
        """
        Close the policy, releasing any resources that it holds (e.g., display windows for plotting).
        """

        super().close()

        if self.action_scatter_plot is not None:
            self.action_scatter_plot.close()

    def __init__(
            self,
            environment: ContinuousMdpEnvironment,
            feature_extractor: StateFeatureExtractor,
            plot_policy: bool
    ):
        """
        Initialize the parameterized policy.

        :param environment: Environment.
        :param feature_extractor: Feature extractor.
        :param plot_policy: Whether or not to plot policy values (e.g., action).
        """

        super().__init__()

        self.environment = environment
        self.feature_extractor = feature_extractor
        self.plot_policy = plot_policy

        self.action_scatter_plot = None
        if self.plot_policy:
            self.action_scatter_plot = ScatterPlot('Actions', self.environment.get_action_dimension_names(), None)

        self.action = None  # we'll fill this in upon the first call to __getitem__, where we have access to a state and its actions.
        self.random_state = RandomState(12345)

    def __contains__(
            self,
            state: MdpState
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

        if state is None:  # pragma no cover
            raise ValueError('Attempted to check for None in policy.')

        return True


@rl_text(chapter=13, page=335)
class ContinuousActionNormalDistributionPolicy(ContinuousActionPolicy):
    """
    Parameterized policy that produces continuous, multi-dimensional actions by modeling the multidimensional mean and
    covariance matrix of the multivariate normal distribution in terms of state features. This is appropriate for action
    spaces that are unbounded in (-infinity, infinity). The state features must be extracted by an extractor derived
    from `rlai.v_S.function_approximation.models.feature_extraction.StateFeatureExtractor`.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: ContinuousMdpEnvironment
    ) -> Tuple[Policy, List[str]]:
        """
        Initialize a policy from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a policy and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # load feature extractor
        feature_extractor_class = load_class(parsed_args.policy_feature_extractor)
        feature_extractor, unparsed_args = feature_extractor_class.init_from_arguments(
            args=unparsed_args,
            environment=environment
        )
        del parsed_args.policy_feature_extractor

        # initialize policy
        policy = cls(
            environment=environment,
            feature_extractor=feature_extractor,
            **vars(parsed_args)
        )

        return policy, unparsed_args

    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

        # extract state-feature matrix
        state_feature_matrix = np.array([
            self.feature_extractor.extract(s, False)
            for s in self.update_batch_s
        ])

        # add intercept
        intercept_state_feature_matrix = np.ones(shape=np.add(state_feature_matrix.shape, (0, 1)))
        intercept_state_feature_matrix[:, 1:] = state_feature_matrix

        action_matrix = np.array([
            a.value
            for a in self.update_batch_a
        ])

        # calculate per-update gradients
        (
            action_density_gradients_wrt_theta_mean,
            action_density_gradients_wrt_theta_cov
        ) = self.get_action_density_gradients_vmap(
            self.theta_mean,
            self.theta_cov,
            intercept_state_feature_matrix,
            action_matrix
        )

        # assemble updates
        updates = zip(
            self.update_batch_a,
            self.update_batch_s,
            intercept_state_feature_matrix,
            self.update_batch_alpha,
            self.update_batch_target,
            action_density_gradients_wrt_theta_mean,
            action_density_gradients_wrt_theta_cov
        )

        for a, s, state_features, alpha, target, action_density_gradient_wrt_theta_mean, action_density_gradient_wrt_theta_cov in updates:

            # TODO:  How to estimate this from the PDF? The multivariate_normal class has a CDF.
            p_a_s = 1.0

            # check for nans in the gradients and skip the update if any are found
            if np.isinf(action_density_gradient_wrt_theta_mean).any() or np.isnan(action_density_gradient_wrt_theta_mean).any() or np.isinf(action_density_gradient_wrt_theta_cov).any() or np.isnan(action_density_gradient_wrt_theta_cov).any():  # pragma no cover
                warnings.warn('Gradients contain np.inf or np.nan value(s). Skipping update.')
            else:

                # check whether the covariance matrix resulting from the updated parameters will be be positive
                # definite, as the multivariate normal distribution requires this. assign the update only if it is so.
                new_theta_cov = self.theta_cov + alpha * target * (action_density_gradient_wrt_theta_cov / p_a_s)
                new_cov = self.get_covariance_matrix(
                    new_theta_cov,
                    state_features
                )

                if is_positive_definite(new_cov):
                    self.theta_mean += alpha * target * (action_density_gradient_wrt_theta_mean / p_a_s)
                    self.theta_cov = new_theta_cov
                else:  # pragma no cover
                    warnings.warn('The updated covariance theta parameters will produce a covariance matrix that is not positive definite. Skipping update.')

    def get_covariance_matrix(
            self,
            theta_cov: np.ndarray,
            state_features: np.ndarray
    ) -> np.ndarray:
        """
        Get covariance matrix from its parameters.

        :param theta_cov: Parameters.
        :param state_features: State features.
        :return: Covariance matrix.
        """

        return np.array([

            # ensure that the diagonal of the covariance matrix has positive values by exponentiating
            np.exp(np.dot(theta_cov_row, state_features)) if i % (self.environment.get_action_space_dimensionality() + 1) == 0

            # off-diagonal elements can be positive or negative
            else np.dot(theta_cov_row, state_features)

            # iteraate over each row of coefficients
            for i, theta_cov_row in enumerate(theta_cov)

        ]).reshape(self.environment.get_action_space_dimensionality(), self.environment.get_action_space_dimensionality())

    @staticmethod
    def get_action_density(
            theta_mean: np.ndarray,
            theta_cov: np.ndarray,
            state_features: np.ndarray,
            action_vector: np.ndarray
    ) -> float:
        """
        Get the value of the probability density function at an action.

        :param theta_mean: Policy parameters for mean.
        :param theta_cov: Policy parameters for covariance matrix.
        :param state_features: A vector of state features.
        :param action_vector: Multi-dimensional action vector.
        :return: Value of the PDF.
        """

        action_space_dimensionality = action_vector.shape[0]

        mean = jnp.dot(theta_mean, state_features)
        cov = jnp.array([

            # ensure that the diagonal of the covariance matrix has positive values by exponentiating
            jnp.exp(jnp.dot(theta_cov_row, state_features)) if i % (action_space_dimensionality + 1) == 0

            # off-diagonal elements can be positive or negative
            else jnp.dot(theta_cov_row, state_features)

            # iteraate over each row of coefficients
            for i, theta_cov_row in enumerate(theta_cov)

        ]).reshape(action_space_dimensionality, action_space_dimensionality)

        return jstats.multivariate_normal.pdf(x=action_vector, mean=mean, cov=cov)

    def __init__(
            self,
            environment: ContinuousMdpEnvironment,
            feature_extractor: StateFeatureExtractor,
            plot_policy: bool
    ):
        """
        Initialize the parameterized policy.

        :param environment: Environment.
        :param feature_extractor: Feature extractor.
        :param plot_policy: Whether or not to plot policy values (e.g., action).
        """

        super().__init__(
            environment=environment,
            feature_extractor=feature_extractor,
            plot_policy=plot_policy
        )

        # coefficients for mean and covariance. these will be initialized upon the first call to the feature extractor
        # within __getitem__.
        self.theta_mean = None
        self.theta_cov = None

        self.get_action_density_gradients = jit(grad(self.get_action_density, argnums=(0, 1)))
        self.get_action_density_gradients_vmap = jit(vmap(self.get_action_density_gradients, in_axes=(None, None, 0, 0)))

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        self.set_action(state)

        intercept_state_features = np.append([1.0], self.feature_extractor.extract(state, True))

        # initialize coefficients for mean and covariance
        if self.theta_mean is None:
            self.theta_mean = np.zeros(shape=(self.environment.get_action_space_dimensionality(), intercept_state_features.shape[0]))

        if self.theta_cov is None:
            self.theta_cov = np.zeros(shape=(self.environment.get_action_space_dimensionality() ** 2, intercept_state_features.shape[0]))

        # calculate the modeled mean and covariance of the n-dimensional action
        mean = self.theta_mean.dot(intercept_state_features)
        cov = self.get_covariance_matrix(
            self.theta_cov,
            intercept_state_features
        )

        # sample the n-dimensional action
        action_value = stats.multivariate_normal.rvs(mean=mean, cov=cov, random_state=self.random_state)

        # convert scalar action to array (e.g., if we're dealing with a 1-dimensional action)
        if np.isscalar(action_value):
            action_value = np.array([action_value])

        action = ContinuousMultiDimensionalAction(
            value=action_value,
            min_values=None,
            max_values=None
        )

        self.update_action_scatter_plot(action)

        return {action: 1.0}

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        state['get_action_density_gradients'] = None
        state['get_action_density_gradients_vmap'] = None

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set unpickled state.

        :param state: Unpickled state.
        """

        get_action_density_gradients = state['get_action_density_gradients'] = jit(grad(self.get_action_density, argnums=(0, 1)))
        state['get_action_density_gradients_vmap'] = jit(vmap(get_action_density_gradients, in_axes=(None, None, 0, 0)))

        self.__dict__ = state

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current policy equals another.

        :param other: Other policy.
        :return: True if policies are equal and False otherwise.
        """

        other: ContinuousActionNormalDistributionPolicy

        # using the default values for allclose is too strict to achieve cross-platform testing success. back off a little with atol.
        return np.allclose(self.theta_mean, other.theta_mean, atol=0.0001) and np.allclose(self.theta_cov, other.theta_cov, atol=0.0001)

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the current policy does not equal another.

        :param other: Other policy.
        :return: True if policies are not equal and False otherwise.
        """

        return not (self == other)


@rl_text(chapter=13, page=335)
class ContinuousActionBetaDistributionPolicy(ContinuousActionPolicy):
    """
    Parameterized policy that produces continuous, multi-dimensional actions by modeling multiple independent beta
    distributions in terms of state features. This is appropriate for action spaces that are bounded in [min, max],
    where the values of min and max can be different along each action dimension. The state features must be extracted
    by an extractor derived from `rlai.v_S.function_approximation.models.feature_extraction.StateFeatureExtractor`.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: ContinuousMdpEnvironment
    ) -> Tuple[Policy, List[str]]:
        """
        Initialize a policy from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a policy and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # load feature extractor
        feature_extractor_class = load_class(parsed_args.policy_feature_extractor)
        feature_extractor, unparsed_args = feature_extractor_class.init_from_arguments(
            args=unparsed_args,
            environment=environment
        )
        del parsed_args.policy_feature_extractor

        # initialize policy
        policy = cls(
            environment=environment,
            feature_extractor=feature_extractor,
            **vars(parsed_args)
        )

        return policy, unparsed_args

    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

        # extract state-feature matrix
        state_feature_matrix = np.array([
            self.feature_extractor.extract(s, False)
            for s in self.update_batch_s
        ])

        # add intercept
        intercept_state_feature_matrix = np.ones(shape=np.add(state_feature_matrix.shape, (0, 1)))
        intercept_state_feature_matrix[:, 1:] = state_feature_matrix

        # invert action values back to [0.0, 1.0] (the domain of the beta distribution)
        action_matrix = np.array([
            self.invert_rescale(a.value)
            for a in self.update_batch_a
        ])

        # perform updates per action, since we model the distribution of each action independently of the other actions.
        # we use the transpose of the action matrix so that each iteration of the for-loop contains all experienced
        # values of the action (in action_i_values).
        for action_i, (action_i_theta_a, action_i_theta_b, action_i_values) in enumerate(zip(self.action_theta_a, self.action_theta_b, action_matrix.T)):

            # calculate per-update gradients for the current action
            (
                action_density_gradients_wrt_theta_a,
                action_density_gradients_wrt_theta_b
            ) = self.get_action_density_gradients_vmap(
                action_i_theta_a,
                action_i_theta_b,
                intercept_state_feature_matrix,
                action_i_values
            )

            # assemble updates
            updates = zip(
                self.update_batch_a,
                self.update_batch_s,
                self.update_batch_alpha,
                self.update_batch_target,
                action_density_gradients_wrt_theta_a,
                action_density_gradients_wrt_theta_b
            )

            for a, s, alpha, target, action_density_gradient_wrt_theta_a, action_density_gradient_wrt_theta_b in updates:

                # TODO:  How to estimate this from the PDF?
                p_a_s = 1.0

                # check for nans in the gradients and skip the update if any are found
                if np.isnan(action_density_gradient_wrt_theta_a).any() or np.isnan(action_density_gradient_wrt_theta_b).any():  # pragma no cover
                    warnings.warn('Gradients contain np.nan value(s). Skipping update.')
                else:

                    # squash gradients into [-1.0, 1.0] to handle scaling issues when actions are chosen at the tails of
                    # the beta distribution where the gradients are very large. this also addresses cases of
                    # positive/negative infinite gradients.
                    self.action_theta_a[action_i, :] += alpha * target * (np.tanh(action_density_gradient_wrt_theta_a) / p_a_s)
                    self.action_theta_b[action_i, :] += alpha * target * (np.tanh(action_density_gradient_wrt_theta_b) / p_a_s)

        self.reset_action_scatter_plot_y_range()

        # only output the hyperparameter table if we have a state-dimension name for each feature. some feature
        # extractors expand the feature space beyond the dimensions, and we don't have a good way to generate names for
        # these extra dimensions. such extractors also tend to generate large feature spaces for which tabular output
        # isn't readable.
        if logging.getLogger().level <= logging.DEBUG and self.action_theta_a.shape[1] == len(self.environment.get_state_dimension_names()) + 1:
            row_names = [
                f'{action_name}_{p}'
                for action_name in self.environment.get_action_dimension_names()
                for p in ['a', 'b']
            ]
            col_names = ['intercept'] + self.environment.get_state_dimension_names()
            theta_df = pd.DataFrame([
                row
                for theta_a_row, theta_b_row in zip(self.action_theta_a, self.action_theta_b)
                for row in [theta_a_row, theta_b_row]
            ], index=row_names, columns=col_names)
            logging.debug(f'Per-action beta hyperparameters:\n{tabulate(theta_df, headers="keys", tablefmt="psql")}')

    def reset_action_scatter_plot_y_range(
            self
    ):
        """
        Reset the y-range in the scatter plot.
        """

        super().reset_action_scatter_plot_y_range()

        if self.beta_shape_scatter_plot is not None:
            self.beta_shape_scatter_plot.reset_y_range()

    @staticmethod
    def get_action_density(
            theta_a: np.ndarray,
            theta_b: np.ndarray,
            state_features: np.ndarray,
            action_value: float
    ) -> float:
        """
        Get the value of the probability density function at an action.

        :param theta_a: Policy parameters for shape parameter a.
        :param theta_b: Policy parameters for shape parameter b.
        :param state_features: A vector of state features.
        :param action_value: Action value.
        :return: Value of the PDF.
        """

        a = jnp.exp(jnp.dot(theta_a, state_features))
        b = jnp.exp(jnp.dot(theta_b, state_features))

        return jstats.beta.pdf(x=action_value, a=a, b=b, loc=0.0, scale=1.0)

    def rescale(
            self,
            action_value: np.ndarray
    ) -> np.ndarray:
        """
        Rescale an action value from [0.0, 1.0] to be in the range expected by the environment.

        :param action_value: Action value.
        :return: Rescaled action value.
        """

        value_ranges = [
            max_value - min_value
            for min_value, max_value in zip(self.action.min_values, self.action.max_values)
        ]

        rescaled_action_value = np.array([
            min_value + value * value_range
            for value, min_value, value_range in zip(action_value, self.action.min_values, value_ranges)
        ])

        return rescaled_action_value

    def invert_rescale(
            self,
            rescaled_action_value: np.ndarray
    ) -> np.ndarray:
        """
        Invert a rescaled action value from its range (expected by the environment) back to [0.0, 1.0] (expected by the
        beta distribution).

        :param rescaled_action_value: Rescaled action value.
        :return: Action value.
        """

        value_ranges = [
            max_value - min_value
            for min_value, max_value in zip(self.action.min_values, self.action.max_values)
        ]

        action_value = np.array([
            (value - min_value) / value_range
            for value, min_value, value_range in zip(rescaled_action_value, self.action.min_values, value_ranges)
        ])

        return action_value

    def close(
            self
    ):
        """
        Close the policy, releasing any resources that it holds (e.g., display windows for plotting).
        """

        super().close()

        if self.beta_shape_scatter_plot is not None:
            self.beta_shape_scatter_plot.close()

    def __init__(
            self,
            environment: ContinuousMdpEnvironment,
            feature_extractor: StateFeatureExtractor,
            plot_policy: bool
    ):
        """
        Initialize the parameterized policy.

        :param environment: Environment.
        :param feature_extractor: Feature extractor.
        :param plot_policy: Whether or not to plot policy values (e.g., action).
        """

        super().__init__(
            environment=environment,
            feature_extractor=feature_extractor,
            plot_policy=plot_policy
        )

        # coefficients for shape parameters a and b. these will be initialized upon the first call to the feature
        # extractor within __getitem__.
        self.action_theta_a = None
        self.action_theta_b = None

        # get jax function for gradients with respect to theta_a and theta_b. vectorize the gradient calculation over
        # input arrays for state and action values.
        self.get_action_density_gradients = jit(grad(self.get_action_density, argnums=(0, 1)))
        self.get_action_density_gradients_vmap = jit(vmap(self.get_action_density_gradients, in_axes=(None, None, 0, 0)))

        self.beta_shape_scatter_plot = None
        if self.plot_policy:
            self.beta_shape_scatter_plot_x_tick_labels = [
                label
                for action_name in self.environment.get_action_dimension_names()
                for label in [f'{action_name} a', f'{action_name} b']
            ]
            self.beta_shape_scatter_plot = ScatterPlot('Beta Distribution Shape', self.beta_shape_scatter_plot_x_tick_labels, None)

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        self.set_action(state)

        intercept_state_features = np.append([1.0], self.feature_extractor.extract(state, True))

        # initialize coefficients for shape parameters a and b
        if self.action_theta_a is None:
            self.action_theta_a = np.zeros(shape=(self.environment.get_action_space_dimensionality(), intercept_state_features.shape[0]))

        if self.action_theta_b is None:
            self.action_theta_b = np.zeros(shape=(self.environment.get_action_space_dimensionality(), intercept_state_features.shape[0]))

        # calculate the modeled shape parameters of each action dimension
        action_a = np.exp(self.action_theta_a.dot(intercept_state_features))
        action_b = np.exp(self.action_theta_b.dot(intercept_state_features))

        # sample each of the action dimensions and rescale
        action_value = self.rescale(
            np.array([
                stats.beta.rvs(a=a, b=b, loc=0.0, scale=1.0, random_state=self.random_state)
                for a, b in zip(action_a, action_b)
            ])
        )

        action = ContinuousMultiDimensionalAction(
            value=action_value,
            min_values=self.action.min_values,
            max_values=self.action.max_values
        )

        if self.plot_policy:
            self.update_action_scatter_plot(action)
            self.beta_shape_scatter_plot.update(np.array([
                v
                for a, b in zip(action_a, action_b)
                for v in [a, b]
            ]))

        return {action: 1.0}

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        state['get_action_density_gradients'] = None
        state['get_action_density_gradients_vmap'] = None

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set unpickled state.

        :param state: Unpickled state.
        """

        get_action_density_gradients = state['get_action_density_gradients'] = jit(grad(self.get_action_density, argnums=(0, 1)))
        state['get_action_density_gradients_vmap'] = jit(vmap(get_action_density_gradients, in_axes=(None, None, 0, 0)))

        self.__dict__ = state

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current policy equals another.

        :param other: Other policy.
        :return: True if policies are equal and False otherwise.
        """

        other: ContinuousActionBetaDistributionPolicy

        # using the default values for allclose is too strict to achieve cross-platform testing success. back off a little with atol.
        return np.allclose(self.action_theta_a, other.action_theta_a, atol=0.0001) and np.allclose(self.action_theta_b, other.action_theta_b, atol=0.0001)

    def __ne__(
            self,
            other
    ) -> bool:
        """
        Check whether the current policy does not equal another.

        :param other: Other policy.
        :return: True if policies are not equal and False otherwise.
        """

        return not (self == other)
