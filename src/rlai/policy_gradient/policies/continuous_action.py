import logging
import warnings
from abc import ABC
from argparse import ArgumentParser
from typing import List, Tuple, Dict, Optional, Callable

import numpy as np
import pandas as pd
from jax import numpy as jnp, jit, grad, vmap, Array
from jax.scipy import stats as jstats
from numpy.random import RandomState
from scipy import stats
from tabulate import tabulate

from rlai.core import Policy, Action, ContinuousMultiDimensionalAction, MdpState
from rlai.core.environments.mdp import ContinuousMdpEnvironment
from rlai.docs import rl_text
from rlai.policy_gradient.policies import ParameterizedPolicy
from rlai.state_value.function_approximation.models.feature_extraction import StateFeatureExtractor
from rlai.utils import parse_arguments, load_class, is_positive_definite


@rl_text(chapter=13, page=335)
class ContinuousActionPolicy(ParameterizedPolicy, ABC):
    """
    Parameterized policy that produces continuous, multidimensional actions.
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
        :param plot_policy: Whether to plot policy values (e.g., action).
        """

        super().__init__()

        self.environment = environment
        self.feature_extractor = feature_extractor
        self.plot_policy = plot_policy

        self.action_scatter_plot = None
        if self.plot_policy:

            # local-import so that we don't crash on raspberry pi os, where we can't install qt6.
            from rlai.plot_utils import ScatterPlot

            self.action_scatter_plot = ScatterPlot(
                'Actions',
                self.environment.get_action_dimension_names(),
                None
            )

        # we'll fill this in upon the first call to __getitem__, where we have access to a state and its actions.
        self.action: Optional[ContinuousMultiDimensionalAction] = None

        self.random_state = RandomState(12345)

    def set_action(
            self,
            state: MdpState
    ):
        """
        Set the single, continuous, multidimensional action for this policy based on a state. This function can be
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
            assert action.value is not None
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

    def reset_for_new_run(
            self,
            state: MdpState
    ):
        """
        Reset for new run.
        """

        self.feature_extractor.reset_for_new_run(state)

    def __contains__(
            self,
            state: Optional[MdpState]
    ) -> bool:
        """
        Check whether the policy is defined for a state.

        :param state: State.
        :return: True if policy is defined for state and False otherwise.
        """

        if state is None:  # pragma no cover
            raise ValueError('Attempted to check for None in policy.')

        return True

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        # certain environments cannot be pickled, and even if all could be pickled they wouldn't be used when running
        # the policy later. this is because we always instantiate a new environment when starting the process. retaining
        # the environment here would cause its references (e.g., to scatter plots) to be invalid.
        state['environment'] = None

        return state


@rl_text(chapter=13, page=335)
class ContinuousActionNormalDistributionPolicy(ContinuousActionPolicy):
    """
    Parameterized policy that produces continuous, multidimensional actions by modeling the multidimensional mean and
    covariance matrix of the multivariate normal distribution in terms of state features. This is appropriate for action
    spaces that are unbounded in (-infinity, infinity). The state features must be extracted by an extractor derived
    from `rlai.state_value.function_approximation.models.feature_extraction.StateFeatureExtractor`.
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

    @staticmethod
    def get_action_density(
            theta_mean: np.ndarray,
            theta_cov: np.ndarray,
            state_features: np.ndarray,
            action_vector: np.ndarray
    ) -> Array:
        """
        Get the value of the probability density function at an action.

        :param theta_mean: Policy parameters for mean.
        :param theta_cov: Policy parameters for covariance matrix.
        :param state_features: A vector of state features.
        :param action_vector: Multidimensional action vector.
        :return: Value of the PDF.
        """

        action_space_dimensionality = action_vector.shape[0]

        mean = jnp.dot(theta_mean, state_features)
        cov = jnp.array([

            # ensure that the diagonal of the covariance matrix has positive values by exponentiating
            jnp.exp(jnp.dot(theta_cov_row, state_features)) if i % (action_space_dimensionality + 1) == 0

            # off-diagonal elements can be positive or negative
            else jnp.dot(theta_cov_row, state_features)

            # iterate over each row of coefficients
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
        :param plot_policy: Whether to plot policy values (e.g., action).
        """

        super().__init__(
            environment=environment,
            feature_extractor=feature_extractor,
            plot_policy=plot_policy
        )

        # coefficients for mean and covariance. these will be initialized upon the first call to the feature extractor
        # within __getitem__.
        self.theta_mean: Optional[np.ndarray] = None
        self.theta_cov: Optional[np.ndarray] = None

        self.get_action_density_gradients = jit(grad(self.get_action_density, argnums=(0, 1)))
        self.get_action_density_gradients_vmap = jit(
            vmap(self.get_action_density_gradients, in_axes=(None, None, 0, 0)))

    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`. Not intended to be called directly
        by outside callers or inheritors.
        """

        # extract state-feature matrix
        state_feature_matrix = self.feature_extractor.extract(self.update_batch_s, True)

        # add intercept if the extractor doesn't extract one
        if not self.feature_extractor.extracts_intercept():
            intercept_state_feature_matrix = np.ones(shape=np.add(state_feature_matrix.shape, (0, 1)))
            intercept_state_feature_matrix[:, 1:] = state_feature_matrix
            state_feature_matrix = intercept_state_feature_matrix

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
            state_feature_matrix,
            action_matrix
        )

        # assemble updates
        updates = zip(
            state_feature_matrix,
            self.update_batch_alpha,
            self.update_batch_target,
            action_density_gradients_wrt_theta_mean,
            action_density_gradients_wrt_theta_cov
        )

        for state_features, alpha, target, action_density_gradient_wrt_theta_mean, action_density_gradient_wrt_theta_cov in updates:

            # check for nans in the gradients and skip the update if any are found
            if (
                    np.isinf(action_density_gradient_wrt_theta_mean).any() or
                    np.isnan(action_density_gradient_wrt_theta_mean).any() or
                    np.isinf(action_density_gradient_wrt_theta_cov).any() or
                    np.isnan(action_density_gradient_wrt_theta_cov).any()
            ):  # pragma no cover
                warnings.warn('Gradients contain np.inf or np.nan value(s). Skipping update.')
            else:

                # check whether the covariance matrix resulting from the updated parameters will be positive
                # definite, as the multivariate normal distribution requires this. assign the update only if it is so.
                new_theta_cov = self.theta_cov + alpha * target * action_density_gradient_wrt_theta_cov
                new_cov = self.get_covariance_matrix(
                    new_theta_cov,
                    state_features
                )

                if is_positive_definite(new_cov):
                    self.theta_mean += alpha * target * action_density_gradient_wrt_theta_mean
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

            # iterate over each row of coefficients
            for i, theta_cov_row in enumerate(theta_cov)

        ]).reshape(self.environment.get_action_space_dimensionality(), self.environment.get_action_space_dimensionality())

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

        state_feature_vector = self.feature_extractor.extract([state], False)[0]

        # add intercept if the extractor doesn't extract one
        if not self.feature_extractor.extracts_intercept():
            state_feature_vector = np.append([1.0], state_feature_vector)

        # initialize coefficients for mean and covariance
        if self.theta_mean is None:
            self.theta_mean = np.zeros(
                shape=(self.environment.get_action_space_dimensionality(), state_feature_vector.shape[0])
            )

        if self.theta_cov is None:
            self.theta_cov = np.zeros(
                shape=(self.environment.get_action_space_dimensionality() ** 2, state_feature_vector.shape[0])
            )

        # calculate the modeled mean and covariance of the n-dimensional action
        mean = self.theta_mean.dot(state_feature_vector)
        cov = self.get_covariance_matrix(
            self.theta_cov,
            state_feature_vector
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

        state = super().__getstate__()

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
            other: object
    ) -> bool:
        """
        Check whether the current policy equals another.

        :param other: Other policy.
        :return: True if policies are equal and False otherwise.
        """

        if not isinstance(other, ContinuousActionNormalDistributionPolicy):
            raise ValueError(f'Expected {ContinuousActionNormalDistributionPolicy}')

        assert self.theta_mean is not None
        assert other.theta_mean is not None
        assert self.theta_cov is not None
        assert other.theta_cov is not None

        # using the default values for `allclose` is too strict to achieve cross-platform testing success. back off a little with atol.
        return np.allclose(self.theta_mean, other.theta_mean, atol=0.0001) and np.allclose(self.theta_cov, other.theta_cov, atol=0.0001)

    def __ne__(
            self,
            other: object
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
    Parameterized policy that produces continuous, multidimensional actions by modeling multiple independent beta
    distributions in terms of state features. This is appropriate for action spaces that are bounded in [min, max],
    where the values of min and max can be different along each action dimension. The state features must be extracted
    by an extractor derived from
    `rlai.state_value.function_approximation.models.feature_extraction.StateFeatureExtractor`.
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

        assert self.action_theta_a is not None
        assert self.action_theta_b is not None

        # extract state-feature matrix:  one row per update and one column per state dimension.
        state_feature_matrix = self.feature_extractor.extract(self.update_batch_s, True)

        # add intercept if the extractor doesn't extract one
        if not self.feature_extractor.extracts_intercept():
            intercept_state_feature_matrix = np.ones(shape=np.add(state_feature_matrix.shape, (0, 1)))
            intercept_state_feature_matrix[:, 1:] = state_feature_matrix
            state_feature_matrix = intercept_state_feature_matrix

        # invert action updates back to [0.0, 1.0] (the domain of the beta distribution), creating one row per action
        # update and one column per action dimension.
        action_matrix = np.array([
            self.invert_rescale(a.value)
            for a in self.update_batch_a
        ])

        # perform updates per action dimension, since we model the distribution of each action independently of the
        # other actions. we use the transpose of the action matrix so that each iteration of the for-loop contains all
        # updates of the action (in action_i_values).
        for (
            action_i,  # action dimension
            (
                action_i_theta_a,  # coefficients (theta) applied to state values to obtain shape parameter a
                action_i_theta_b,  # coefficients (theta) applied to state values to obtain shape parameter b
                action_i_values  # action update values
            )
        ) in enumerate(
            zip(
                self.action_theta_a,
                self.action_theta_b,
                action_matrix.T
            )
        ):
            # calculate per-update gradients for the current action dimension with respect to the action's policy
            # parameters. the following call is vectorized over the rows of the state-feature matrix and the
            # action_i_values. each of the return values (action_density_gradients_wrt_theta_a and
            # action_density_gradients_wrt_theta_b) will have one row per update, and each such row will be a vector of
            # partial gradients for the associated state features and action.
            (
                action_density_gradients_wrt_theta_a,
                action_density_gradients_wrt_theta_b
            ) = self.get_action_density_gradients_vmap(
                action_i_theta_a,
                action_i_theta_b,
                state_feature_matrix,
                action_i_values
            )

            # update the theta-a and theta-b coefficients for the current action dimension
            for (
                alpha,
                target,
                action_density_gradient_wrt_theta_a,
                action_density_gradient_wrt_theta_b
            ) in zip(
                self.update_batch_alpha,
                self.update_batch_target,
                action_density_gradients_wrt_theta_a,
                action_density_gradients_wrt_theta_b
            ):

                # step the theta-a and theta-b coefficient vectors in the direction of the target according to the
                # action-density gradients. a positive target will result in more density around the updated action, and
                # a negative target will result in less density around the updated action. a few details:
                #
                #   * normalize the action-density gradients to be unit length to avoid wild derivatives at the tails
                #     of the beta distribution.
                #   * check for any near-zero elements in the gradient norm, and do not update if they exist (must not
                #     divide by zero).
                #   * use nan-to-num to avoid the use of infinite gradients. any infinite value in the gradient will
                #     cause the entire gradient norm to be zero, resulting in no update.

                target_step = alpha * target

                action_density_gradient_wrt_theta_a_norm = np.linalg.norm(action_density_gradient_wrt_theta_a)
                if not np.isclose(action_density_gradient_wrt_theta_a_norm, 0.0):
                    action_density_gradient_wrt_theta_a_unit_length = np.nan_to_num(
                        action_density_gradient_wrt_theta_a / action_density_gradient_wrt_theta_a_norm
                    )
                    self.action_theta_a[action_i, :] += target_step * action_density_gradient_wrt_theta_a_unit_length

                action_density_gradient_wrt_theta_b_norm = np.linalg.norm(action_density_gradient_wrt_theta_b)
                if not np.isclose(action_density_gradient_wrt_theta_b_norm, 0.0):
                    action_density_gradient_wrt_theta_b_unit_length = np.nan_to_num(
                        action_density_gradient_wrt_theta_b / action_density_gradient_wrt_theta_b_norm
                    )
                    self.action_theta_b[action_i, :] += target_step * action_density_gradient_wrt_theta_b_unit_length

        self.reset_action_scatter_plot_y_range()

        if logging.getLogger().level <= logging.INFO:
            action_min_a = np.amin(self.action_theta_a, 1)
            action_max_a = np.amax(self.action_theta_a, 1)
            action_min_b = np.amin(self.action_theta_b, 1)
            action_max_b = np.amax(self.action_theta_b, 1)
            logging.info(f'State dimensions:  {self.action_theta_a.shape[1]}')
            for i, (min_a, max_a, min_b, max_b) in enumerate(zip(action_min_a, action_max_a, action_min_b, action_max_b)):
                logging.info(f'Action {i} [min,max]:\n\ta:  [{min_a},{max_a}]\n\tb:  [{min_b},{max_b}]\n')

        # only output the hyperparameter table if we have a state-dimension name for each feature. some feature
        # extractors expand the feature space beyond the dimension names (e.g., via interaction terms), and we don't
        # have a good way to generate names for these extra features. such extractors also tend to generate large
        # feature spaces for which tabular output isn't easily readable.
        if (
            logging.getLogger().level <= logging.INFO and
            self.action_theta_a.shape[1] == len(self.environment.get_state_dimension_names()) + 1
        ):
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
            logging.info(
                'Per-action beta hyperparameters:\n'
                f'{tabulate(theta_df, headers="keys", tablefmt="psql")}'  # type: ignore[arg-type]
            )

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
    ) -> Array:
        """
        Get the value of the probability density function at an action value.

        :param theta_a: Policy parameters for beta-distribution shape parameter `a`.
        :param theta_b: Policy parameters for beta-distribution shape parameter `b`.
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

        assert self.action is not None
        assert self.action.min_values is not None
        assert self.action.max_values is not None

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

        assert self.action is not None
        assert self.action.min_values is not None
        assert self.action.max_values is not None

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
        :param plot_policy: Whether to plot policy values (e.g., action).
        """

        super().__init__(
            environment=environment,
            feature_extractor=feature_extractor,
            plot_policy=plot_policy
        )

        # coefficients for shape parameters a and b. each array has one row per action and one column per state
        # dimension. these will be initialized upon the first call to the feature extractor within __getitem__.
        self.action_theta_a: Optional[np.ndarray] = None
        self.action_theta_b: Optional[np.ndarray] = None

        # get jax function for gradients with respect to theta_a and theta_b
        self.get_action_density_gradients = jit(grad(fun=self.get_action_density, argnums=(0, 1)))

        # vectorize the gradient calculation over rows (input axis 0) of the input arrays for state and action values.
        # we do not map over the first two positional arguments, which are the theta-a and theta-b coefficient vectors
        # at which we want the gradients.
        self.get_action_density_gradients_vmap = jit(vmap(
            fun=self.get_action_density_gradients,
            in_axes=(None, None, 0, 0)
        ))

        self.beta_shape_scatter_plot = None
        if self.plot_policy:

            # local-import so that we don't crash on raspberry pi os, where we can't install qt6.
            from rlai.plot_utils import ScatterPlot

            self.beta_shape_scatter_plot_x_tick_labels = [
                label
                for action_name in self.environment.get_action_dimension_names()
                for label in [f'{action_name} a', f'{action_name} b']
            ]
            self.beta_shape_scatter_plot = ScatterPlot(
                'Beta Distribution Shape',
                self.beta_shape_scatter_plot_x_tick_labels,
                None
            )

        self.get_item_hook: Optional[Callable] = None

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

        assert self.action is not None

        state_feature_vector = self.feature_extractor.extract([state], False)[0]

        # add intercept if the extractor doesn't extract one
        if not self.feature_extractor.extracts_intercept():
            state_feature_vector = np.append([1.0], state_feature_vector)

        # initialize coefficients for each action's a-shape parameter
        if self.action_theta_a is None:
            self.action_theta_a = np.zeros(
                shape=(self.environment.get_action_space_dimensionality(), state_feature_vector.shape[0])
            )

        # initialize coefficients for each action's b-shape parameter
        if self.action_theta_b is None:
            self.action_theta_b = np.zeros(
                shape=(self.environment.get_action_space_dimensionality(), state_feature_vector.shape[0])
            )

        # calculate the modeled shape parameters of each action dimension
        action_a = np.exp(self.action_theta_a.dot(state_feature_vector))
        action_b = np.exp(self.action_theta_b.dot(state_feature_vector))

        try:

            # sample each of the action dimensions and rescale
            action_value = self.rescale(
                np.array([
                    stats.beta.rvs(a=a, b=b, loc=0.0, scale=1.0, random_state=self.random_state)
                    for a, b in zip(action_a, action_b)
                ])
            )

        # watch out for numerical issues (e.g., alpha step sizes that are too large and create issues for beta
        # sampling). set a uniformly random action in such cases and report the error.
        except ValueError as e:
            if str(e) == 'Domain error in arguments.':
                action_value = self.rescale(
                    np.array([
                        stats.beta.rvs(a=a, b=b, loc=0.0, scale=1.0, random_state=self.random_state)
                        for a, b in zip(np.ones_like(action_a), np.ones_like(action_b))
                    ])
                )
                logging.error(f'Caught {e} while setting action to {action_value}.')
            else:
                raise e

        action = ContinuousMultiDimensionalAction(
            value=action_value,
            min_values=self.action.min_values,
            max_values=self.action.max_values,
            name=self.action.name
        )

        if self.plot_policy:
            self.update_action_scatter_plot(action)
            assert self.beta_shape_scatter_plot is not None
            self.beta_shape_scatter_plot.update(np.array([
                v
                for a, b in zip(action_a, action_b)
                for v in [a, b]
            ]))

        # send the results to the hook if we have one
        if self.get_item_hook is not None:
            self.get_item_hook(
                state_feature_vector,
                action_a,
                action_b,
                action
            )

        return {action: 1.0}

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: State dictionary.
        """

        state = super().__getstate__()

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
            other: object
    ) -> bool:
        """
        Check whether the current policy equals another.

        :param other: Other policy.
        :return: True if policies are equal and False otherwise.
        """

        if not isinstance(other, ContinuousActionBetaDistributionPolicy):
            raise ValueError(f'Expected {ContinuousActionBetaDistributionPolicy}')

        assert self.action_theta_a is not None
        assert other.action_theta_a is not None
        assert self.action_theta_b is not None
        assert other.action_theta_b is not None

        # using the default values for `allclose` is too strict to achieve cross-platform testing success. back off a little with atol.
        return np.allclose(self.action_theta_a, other.action_theta_a, atol=0.0001) and np.allclose(self.action_theta_b, other.action_theta_b, atol=0.0001)

    def __ne__(
            self,
            other: object
    ) -> bool:
        """
        Check whether the current policy does not equal another.

        :param other: Other policy.
        :return: True if policies are not equal and False otherwise.
        """

        return not (self == other)
