import warnings
from abc import ABC
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional

import numpy as np
from jax import numpy as jnp, jit, grad, vmap
from jax.scipy import stats as jstats
from numpy.random import RandomState
from scipy import stats

from rlai.actions import Action, ContinuousMultiDimensionalAction
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.policies.parameterized import ParameterizedPolicy
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments, load_class, is_positive_definite, ScatterPlot, ScatterPlotPosition
from rlai.v_S.function_approximation.models.feature_extraction import StateFeatureExtractor


@rl_text(chapter=13, page=335)
class ContinuousActionDistributionPolicy(ParameterizedPolicy, ABC):
    """
    Parameterized policy that produces continuous, multi-dimensional actions by modeling a multi-dimensional
    distribution in terms of state features. The state features must be extracted by an extractor derived from
    `rlai.v_S.function_approximation.models.feature_extraction.StateFeatureExtractor`.
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

    def __init__(
            self,
            environment: MdpEnvironment,
            feature_extractor: StateFeatureExtractor,
            plot_policy: bool,
            scatter_plot_labels: Optional[List[str]] = None
    ):
        """
        Initialize the parameterized policy.

        :param environment: Environment.
        :param feature_extractor: Feature extractor.
        :param plot_policy: Whether or not to plot policy values (e.g., action).
        :param scatter_plot_labels: Scatter plot labels, to be added to those for the action.
        """

        if scatter_plot_labels is None:
            scatter_plot_labels = []

        super().__init__()

        # TODO: This is poorly abstracted -- only gym environments have such an action
        self.action = environment.actions[0]

        self.feature_extractor = feature_extractor
        self.state_space_dimensionality = self.feature_extractor.get_state_space_dimensionality()
        self.action_space_dimensionality = self.feature_extractor.get_action_space_dimensionality()

        self.plot_policy = plot_policy
        if self.plot_policy:
            self.scatter_plot_x_tick_labels = scatter_plot_labels + [f'Action {i}' for i in range(self.action_space_dimensionality)]
            self.scatter_plot = ScatterPlot('Action', self.scatter_plot_x_tick_labels, ScatterPlotPosition.TOP_RIGHT)

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


class ContinuousActionNormalDistributionPolicy(ContinuousActionDistributionPolicy):
    """
    Parameterized policy that produces continuous, multi-dimensional actions by modeling the multidimensional mean and
    covariance matrix of the multivariate normal distribution in terms of state features. This is appropriate for action
    spaces that are unbounded in (-infinity, infinity).The state features must be extracted by an extractor derived from
    `rlai.v_S.function_approximation.models.feature_extraction.StateFeatureExtractor`.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
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
            np.exp(np.dot(theta_cov_row, state_features)) if i % (self.action_space_dimensionality + 1) == 0

            # off-diagonal elements can be positive or negative
            else np.dot(theta_cov_row, state_features)

            # iteraate over each row of coefficients
            for i, theta_cov_row in enumerate(theta_cov)

        ]).reshape(self.action_space_dimensionality, self.action_space_dimensionality)

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
            environment: MdpEnvironment,
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
        self.theta_mean = np.zeros(shape=(self.action_space_dimensionality, self.state_space_dimensionality + 1))

        # coefficients for multi-dimensional covariance:  one row per entry in the covariance matrix (a square matrix
        # with each action along each dimension) and one column per state feature (plus 1 for the bias/intercept).
        self.theta_cov = np.zeros(shape=(self.action_space_dimensionality ** 2, self.state_space_dimensionality + 1))

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
            self.scatter_plot.update(action_value)

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


class ContinuousActionBetaDistributionPolicy(ContinuousActionDistributionPolicy):
    """
    Parameterized policy that produces continuous, multi-dimensional actions by modeling multiple independent beta
    distributions in terms of state features. This is appropriate for action spaces that are bounded in [min, max]. The
    state features must be extracted by an extractor derived from
    `rlai.v_S.function_approximation.models.feature_extraction.StateFeatureExtractor`.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
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
        for action_i, (action_theta_a, action_theta_b, action_values) in enumerate(zip(self.action_theta_a, self.action_theta_b, action_matrix.T)):

            # calculate per-update gradients for the current action
            action_density_gradients_wrt_theta_a, action_density_gradients_wrt_theta_b = self.get_action_density_gradients_vmap(
                action_theta_a,
                action_theta_b,
                intercept_state_feature_matrix,
                action_values
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

                # TODO:  This is always 1. How to estimate it from the PDF?
                p_a_s = self[s][a]

                # check for nans in the gradients and skip the update if any are found
                if np.isnan(action_density_gradient_wrt_theta_a).any() or np.isnan(action_density_gradient_wrt_theta_b).any():
                    warnings.warn('Gradient contain np.nan value(s). Skipping update.')
                else:
                    self.action_theta_a[action_i, :] += alpha * discounted_return * (action_density_gradient_wrt_theta_a / p_a_s)
                    self.action_theta_b[action_i, :] += alpha * discounted_return * (action_density_gradient_wrt_theta_b / p_a_s)

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

        a = 1.0 + jnp.exp(jnp.dot(theta_a, state_features))
        b = 1.0 + jnp.exp(jnp.dot(theta_b, state_features))

        return jstats.beta.pdf(x=action_value, a=a, b=b, loc=0.0, scale=1.0)

    def rescale(
            self,
            action_value: np.ndarray
    ) -> np.ndarray:

        value_ranges = [
            max_value - min_value
            for min_value, max_value in zip(self.action.min_values, self.action.max_values)
        ]

        action_value = np.array([
            min_value + value * value_range
            for value, min_value, value_range in zip(action_value, self.action.min_values, value_ranges)
        ])

        return action_value

    def rescale_inv(
            self,
            action_value: np.ndarray
    ) -> np.ndarray:

        value_ranges = [
            max_value - min_value
            for min_value, max_value in zip(self.action.min_values, self.action.max_values)
        ]

        action_value = np.array([
            (value - min_value) / value_range
            for value, min_value, value_range in zip(action_value, self.action.min_values, value_ranges)
        ])

        return action_value

    def __init__(
            self,
            environment: MdpEnvironment,
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
            plot_policy=plot_policy,
            scatter_plot_labels=[
                label
                for i in range(feature_extractor.get_action_space_dimensionality())
                for label in [f'Action {i} a', f'Action {i} b']
            ]
        )

        # coefficients for shape parameters a and b:  one row per action and one column per state feature (plus 1 for
        # the bias/intercept).
        self.action_theta_a = np.zeros(shape=(self.action_space_dimensionality, self.state_space_dimensionality + 1))
        self.action_theta_b = np.zeros(shape=(self.action_space_dimensionality, self.state_space_dimensionality + 1))

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

        intercept_state_features = np.append([1.0], self.feature_extractor.extract(state))

        # calculate the modeled shape parameters of the n-dimensional action
        a_values = 1.0 + np.exp(self.action_theta_a.dot(intercept_state_features))
        b_values = 1.0 + np.exp(self.action_theta_b.dot(intercept_state_features))

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
            plot_values = np.append(np.array([
                v
                for a, b in zip(a_values, b_values)
                for v in [a, b]
            ]), action_value)
            self.scatter_plot.update(plot_values)

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
