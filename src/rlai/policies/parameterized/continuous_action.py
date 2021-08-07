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
from rlai.utils import parse_arguments, load_class, is_positive_definite, ScatterPlot, ScatterPlotPosition
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
            help='Pass this flag to plot policy values (e.g., action).'
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
            else:
                raise ValueError('Expected state to contain a single action of type ContinuousMultiDimensionalAction.')

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
            self.action_scatter_plot = ScatterPlot('Actions', self.environment.get_action_dimension_names(), ScatterPlotPosition.TOP_RIGHT)

        self.action = None  # we'll fill this in upon the first call to __getitem__, where we have access to a state and its actions
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

        if state is None:
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
            self.feature_extractor.extract(s)
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
        action_density_gradients_wrt_theta_mean, action_density_gradients_wrt_theta_cov = self.get_action_density_gradients_vmap(
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
            self.update_batch_discounted_return,
            action_density_gradients_wrt_theta_mean,
            action_density_gradients_wrt_theta_cov
        )

        for a, s, state_features, alpha, discounted_return, action_density_gradient_wrt_theta_mean, action_density_gradient_wrt_theta_cov in updates:

            # TODO:  This is always 1. How to estimate it from the PDF? The multivariate_normal class has a CDF.
            p_a_s = self[s][a]

            # check for nans in the gradients and skip the update if any are found
            if np.isnan(action_density_gradient_wrt_theta_mean).any() or np.isnan(action_density_gradient_wrt_theta_cov).any():
                warnings.warn('Gradients contain np.nan value(s). Skipping update.')
            else:

                # check whether the covariance matrix resulting from the updated parameters will be be positive
                # definite, as the multivariate normal distribution requires this. assign the update only if it is so.
                new_theta_cov = self.theta_cov + alpha * discounted_return * (action_density_gradient_wrt_theta_cov / p_a_s)
                new_cov = self.get_covariance_matrix(
                    new_theta_cov,
                    state_features
                )

                if is_positive_definite(new_cov):
                    self.theta_mean += alpha * discounted_return * (action_density_gradient_wrt_theta_mean / p_a_s)
                    self.theta_cov = new_theta_cov
                else:
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

        # coefficients for multi-dimensional mean:  one row per action and one column per state feature (plus 1 for the
        # bias/intercept).
        self.theta_mean = np.zeros(shape=(self.environment.get_action_space_dimensionality(), self.environment.get_state_space_dimensionality() + 1))

        # coefficients for multi-dimensional covariance:  one row per entry in the covariance matrix (a square matrix
        # with each action along each dimension) and one column per state feature (plus 1 for the bias/intercept).
        self.theta_cov = np.zeros(shape=(self.environment.get_action_space_dimensionality() ** 2, self.environment.get_state_space_dimensionality() + 1))

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

        intercept_state_features = np.append([1.0], self.feature_extractor.extract(state))

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

        a = ContinuousMultiDimensionalAction(
            value=action_value,
            min_values=None,
            max_values=None
        )

        if self.plot_policy:
            self.action_scatter_plot.update(action_value)

        return {a: 1.0}

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: Dictionary.
        """

        state_dict = dict(self.__dict__)
        state_dict['get_action_density_gradients'] = None
        state_dict['get_action_density_gradients_vmap'] = None

        return state_dict

    def __setstate__(
            self,
            state_dict: Dict
    ):
        """
        Set unpickled state.

        :param state_dict: Unpickled state.
        """

        state_dict['get_action_density_gradients'] = jit(grad(self.get_action_density, argnums=(0, 1)))
        state_dict['get_action_density_gradients_vmap'] = jit(vmap(self.get_action_density_gradients, in_axes=(None, None, 0, 0)))

        self.__dict__ = state_dict


@rl_text(chapter=13, page=335)
class ContinuousActionBetaDistributionPolicy(ContinuousActionPolicy):
    """
    Parameterized policy that produces continuous, multi-dimensional actions by modeling multiple independent beta
    distributions in terms of state features. This is appropriate for action spaces that are bounded in [min, max],
    where the values of min and max can be different along each action dimension. The state features must be extracted
    by an extractor derived from `rlai.v_S.function_approximation.models.feature_extraction.StateFeatureExtractor`.
    """

    MAX_BETA_SHAPE_VALUE = 50.0

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
            self.feature_extractor.extract(s)
            for s in self.update_batch_s
        ])

        # add intercept
        intercept_state_feature_matrix = np.ones(shape=np.add(state_feature_matrix.shape, (0, 1)))
        intercept_state_feature_matrix[:, 1:] = state_feature_matrix

        # invert actions back to [0.0, 1.0] (the domain of the beta distribution)
        action_matrix = np.array([
            self.rescale_inv(a.value)
            for a in self.update_batch_a
        ])

        # perform updates per-action, since we model the distribution of each action independently of the other actions.
        for action_i, (action_i_theta_a, action_i_theta_b, action_i_values) in enumerate(zip(self.action_theta_a, self.action_theta_b, action_matrix.T)):

            # calculate per-update gradients for the current action
            action_density_gradients_wrt_theta_a, action_density_gradients_wrt_theta_b = self.get_action_density_gradients_vmap(
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
                self.update_batch_discounted_return,
                action_density_gradients_wrt_theta_a,
                action_density_gradients_wrt_theta_b
            )

            for a, s, alpha, discounted_return, action_density_gradient_wrt_theta_a, action_density_gradient_wrt_theta_b in updates:

                # TODO:  This is always 1 but should be calculated per the text. How to estimate it from the PDF?
                p_a_s = self[s][a]

                # check for nans in the gradients and skip the update if any are found
                if np.isnan(action_density_gradient_wrt_theta_a).any() or np.isnan(action_density_gradient_wrt_theta_b).any():
                    warnings.warn('Gradients contain np.nan value(s). Skipping update.')
                else:
                    self.action_theta_a[action_i, :] += alpha * discounted_return * (action_density_gradient_wrt_theta_a / p_a_s)
                    self.action_theta_b[action_i, :] += alpha * discounted_return * (action_density_gradient_wrt_theta_b / p_a_s)

        if self.plot_policy:
            self.action_scatter_plot.reset_y_range()
            self.beta_shape_scatter_plot.reset_y_range()

        if logging.getLogger().level <= logging.DEBUG:
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

        a_x = jnp.dot(theta_a, state_features)
        a = 1.0 + (1.0 / (1.0 + jnp.exp(-a_x))) * ContinuousActionBetaDistributionPolicy.MAX_BETA_SHAPE_VALUE

        b_x = jnp.dot(theta_b, state_features)
        b = 1.0 + (1.0 / (1.0 + jnp.exp(-b_x))) * ContinuousActionBetaDistributionPolicy.MAX_BETA_SHAPE_VALUE

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

    def rescale_inv(
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

        # coefficients for shape parameters a and b:  one row per action and one column per state feature (plus 1 for
        # the bias/intercept).
        self.action_theta_a = np.zeros(shape=(self.environment.get_action_space_dimensionality(), self.environment.get_state_space_dimensionality() + 1))
        self.action_theta_b = np.zeros(shape=(self.environment.get_action_space_dimensionality(), self.environment.get_state_space_dimensionality() + 1))

        self.get_action_density_gradients = jit(grad(self.get_action_density, argnums=(0, 1)))
        self.get_action_density_gradients_vmap = jit(vmap(self.get_action_density_gradients, in_axes=(None, None, 0, 0)))

        self.beta_shape_scatter_plot = None
        if self.plot_policy:
            self.beta_shape_scatter_plot_x_tick_labels = [
                label
                for action_name in self.environment.get_action_dimension_names()
                for label in [f'{action_name} a', f'{action_name} b']
            ]
            self.beta_shape_scatter_plot = ScatterPlot('Beta Distribution Shape', self.beta_shape_scatter_plot_x_tick_labels, ScatterPlotPosition.BOTTOM_RIGHT)

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

        intercept_state_features = np.append([1.0], self.feature_extractor.extract(state))

        # calculate the modeled shape parameters of the n-dimensional action
        a_x_values = self.action_theta_a.dot(intercept_state_features)
        a_values = 1.0 + (1.0 / (1.0 + np.exp(-a_x_values))) * (ContinuousActionBetaDistributionPolicy.MAX_BETA_SHAPE_VALUE - 1.0)

        b_x_values = self.action_theta_b.dot(intercept_state_features)
        b_values = 1.0 + (1.0 / (1.0 + np.exp(-b_x_values))) * (ContinuousActionBetaDistributionPolicy.MAX_BETA_SHAPE_VALUE - 1.0)

        # sample each of the n dimensions and then rescale
        action_value = np.array([
            stats.beta.rvs(a=a, b=b, loc=0.0, scale=1.0, random_state=self.random_state)
            for a, b in zip(a_values, b_values)
        ])

        action_value = self.rescale(action_value)

        a = ContinuousMultiDimensionalAction(
            value=action_value,
            min_values=self.action.min_values,
            max_values=self.action.max_values
        )

        if self.plot_policy:
            self.action_scatter_plot.update(action_value)
            self.beta_shape_scatter_plot.update(np.array([
                v
                for a, b in zip(a_values, b_values)
                for v in [a, b]
            ]))

        return {a: 1.0}

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: Dictionary.
        """

        state_dict = dict(self.__dict__)
        state_dict['get_action_density_gradients'] = None
        state_dict['get_action_density_gradients_vmap'] = None

        return state_dict

    def __setstate__(
            self,
            state_dict: Dict
    ):
        """
        Set unpickled state.

        :param state_dict: Unpickled state.
        """

        state_dict['get_action_density_gradients'] = jit(grad(self.get_action_density, argnums=(0, 1)))
        state_dict['get_action_density_gradients_vmap'] = jit(vmap(self.get_action_density_gradients, in_axes=(None, None, 0, 0)))

        self.__dict__ = state_dict