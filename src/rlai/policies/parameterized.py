import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from functools import reduce
from typing import Dict, List, Tuple

import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
from jax import grad, jit, vmap
from numpy.random import RandomState
from scipy import stats

from rlai.actions import Action, ContinuousMultiDimensionalAction
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.models.feature_extraction import NonstationaryFeatureScaler
from rlai.policies import Policy
from rlai.q_S_A.function_approximation.models import FeatureExtractor
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments, load_class, get_base_argument_parser, is_positive_definite
from rlai.v_S.function_approximation.models.feature_extraction import StateFeatureExtractor


@rl_text(chapter=13, page=321)
class ParameterizedPolicy(Policy, ABC):
    """
    Policy for use with policy gradient methods.
    """

    @classmethod
    def get_argument_parser(
            cls
    ) -> ArgumentParser:
        """
        Get argument parser.

        :return: Argument parser.
        """

        return get_base_argument_parser()

    def append_update(
            self,
            a: Action,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ):
        """
        Append an update for an action in a state using a discounted return and a step size. All appended updates will
        be commited to the policy when `commit_updates` is called.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        """

        self.update_batch_a.append(a)
        self.update_batch_s.append(s)
        self.update_batch_alpha.append(alpha)
        self.update_batch_discounted_return.append(discounted_return)

    def commit_updates(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

        self.__commit_updates__()

        self.update_batch_a.clear()
        self.update_batch_s.clear()
        self.update_batch_alpha.clear()
        self.update_batch_discounted_return.clear()

    @abstractmethod
    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

    def __init__(
            self
    ):
        """
        Initialize the parameterized policy.
        """

        self.update_batch_a = []
        self.update_batch_s = []
        self.update_batch_alpha = []
        self.update_batch_discounted_return = []


@rl_text(chapter=13, page=322)
class SoftMaxInActionPreferencesPolicy(ParameterizedPolicy):
    """
    Parameterized policy that implements a soft-max over action preferences. The policy gradient calculation is coded up
    manually. See the `JaxSoftMaxInActionPreferencesPolicy` for a similar policy in which the gradient is calculated
    using the JAX library. This is only compatible with feature extractors derived from
    `rlai.q_S_A.function_approximation.models.feature_extraction.FeatureExtractor`, which returns state-action feature
    vectors.
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

        return parser

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

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        # initialize policy
        policy = cls(
            feature_extractor=feature_extractor
        )

        return policy, unparsed_args

    def gradient(
            self,
            a: Action,
            s: MdpState
    ) -> np.ndarray:
        """
        Calculate the gradient of the policy for an action in a state, with respect to the policy's parameter vector.

        :param a: Action.
        :param s: State.
        :return: Vector of partial gradients, one per parameter.
        """

        # numerator and its gradient
        x_s_a = self.feature_extractor.extract([s], [a], True)[0, :]
        soft_max_numerator = np.exp(self.theta.dot(x_s_a))
        gradient_soft_max_numerator = soft_max_numerator * x_s_a

        # denominator's state-action feature vectors
        x_s_aa_list = [
            self.feature_extractor.extract([s], [aa], True)[0, :]
            for aa in s.AA
        ]

        # denominator's addends
        soft_max_denominator_addends = np.array([
            np.exp(self.theta.dot(x_s_aa))
            for x_s_aa in x_s_aa_list
        ])

        # denominator
        soft_max_denominator = soft_max_denominator_addends.sum()

        # denominator's gradient
        gradient_soft_max_denominator = reduce(
            np.add,
            [
                addend * x_s_aa
                for addend, x_s_aa in zip(soft_max_denominator_addends, x_s_aa_list)
            ]
        )

        # quotient rule for policy gradient
        gradient = (soft_max_denominator * gradient_soft_max_numerator - soft_max_numerator * gradient_soft_max_denominator) / (soft_max_denominator ** 2.0)

        return gradient

    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

        updates = zip(
            self.update_batch_a,
            self.update_batch_s,
            self.update_batch_alpha,
            self.update_batch_discounted_return
        )

        for a, s, alpha, discounted_return in updates:
            gradient_a_s = self.gradient(a, s)
            p_a_s = self[s][a]
            self.theta += alpha * discounted_return * (gradient_a_s / p_a_s)

    def __init__(
            self,
            feature_extractor: FeatureExtractor
    ):
        """
        Initialize the parameterized policy.

        :param feature_extractor: Feature extractor.
        """

        super().__init__()

        self.feature_extractor = feature_extractor

        self.theta = np.zeros(sum(len(features) for features in feature_extractor.get_action_feature_names().values()))

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

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        soft_max_denominator_addends = np.array([
            np.exp(self.theta.dot(self.feature_extractor.extract([state], [a], False)[0, :]))
            for a in state.AA
        ])

        soft_max_denominator = soft_max_denominator_addends.sum()

        action_prob = {
            a: soft_max_denominator_addends[i] / soft_max_denominator
            for i, a in enumerate(state.AA)
        }

        return action_prob


@rl_text(chapter=13, page=322)
class SoftMaxInActionPreferencesJaxPolicy(ParameterizedPolicy):
    """
    Parameterized policy that implements a soft-max over action preferences. The policy gradient calculation is
    performed using the JAX library. This is only compatible with feature extractors derived from
    `rlai.q_S_A.function_approximation.models.feature_extraction.FeatureExtractor`, which returns state-action feature
    vectors.
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

        return parser

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

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        # initialize policy
        policy = cls(
            feature_extractor=feature_extractor
        )

        return policy, unparsed_args

    def gradient(
            self,
            a: Action,
            s: MdpState
    ) -> np.ndarray:
        """
        Calculate the gradient of the policy for an action in a state, with respect to the policy's parameter vector.

        :param a: Action.
        :param s: State.
        :return: Vector of partial gradients, one per parameter.
        """

        state_action_features = self.get_state_action_features(s)
        gradient = self.get_action_prob_gradient(self.theta, state_action_features, a.i)

        return gradient

    def get_state_action_features(
            self,
            state: MdpState
    ) -> np.ndarray:
        """
        Get a matrix containing a feature vector for each action in a state.

        :param state: State.
        :return: An (#features, #actions) matrix.
        """

        return np.array([
            self.feature_extractor.extract([state], [a], False)[0, :]
            for a in state.AA
        ]).transpose()

    @staticmethod
    def get_action_prob(
            theta: np.ndarray,
            state_action_features: np.ndarray,
            action_i: int
    ) -> float:
        """
        Get action probability.

        :param theta: Policy parameters.
        :param state_action_features: An (#features, #actions) matrix, as returned by `get_state_action_features`.
        :param action_i: Index of action to get probability for.
        :return: Action probability.
        """

        soft_max_denominator_addends = jnp.exp(jnp.dot(theta, state_action_features))
        soft_max_denominator = soft_max_denominator_addends.sum()

        return soft_max_denominator_addends[action_i] / soft_max_denominator

    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

        updates = zip(
            self.update_batch_a,
            self.update_batch_s,
            self.update_batch_alpha,
            self.update_batch_discounted_return
        )

        for a, s, alpha, discounted_return in updates:
            gradient_a_s = self.gradient(a, s)
            p_a_s = self[s][a]
            self.theta += alpha * discounted_return * (gradient_a_s / p_a_s)

    def __init__(
            self,
            feature_extractor: FeatureExtractor
    ):
        """
        Initialize the parameterized policy.

        :param feature_extractor: Feature extractor.
        """

        super().__init__()

        self.feature_extractor = feature_extractor

        self.theta = np.zeros(sum(len(features) for features in feature_extractor.get_action_feature_names().values()))
        self.get_action_prob_gradient = grad(self.get_action_prob)

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

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        state_action_features = self.get_state_action_features(state)

        action_prob = {
            a: self.get_action_prob(self.theta, state_action_features, i)
            for i, a in enumerate(state.AA)
        }

        return action_prob

    def __getstate__(
            self
    ) -> Dict:
        """
        Get state dictionary for pickling.

        :return: Dictionary.
        """

        state_dict = dict(self.__dict__)
        state_dict['get_action_prob_gradient'] = None

        return state_dict

    def __setstate__(
            self,
            state_dict: Dict
    ):
        """
        Set unpickled state.

        :param state_dict: Unpickled state.
        """

        state_dict['get_action_prob_gradient'] = grad(self.get_action_prob)

        self.__dict__ = state_dict


@rl_text(chapter=13, page=335)
class ContinuousActionDistributionPolicy(ParameterizedPolicy):
    """
    Parameterized policy that produces continuous, multi-dimensional actions by modeling a multi-dimensional
    distribution (e.g., the multidimensional mean and covariance matrix of the multivariate normal distribution) in
    terms of state features. The state features must be extracted by an extractor derived from
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

        return parser

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

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        # initialize policy
        policy = cls(
            feature_extractor=feature_extractor
        )

        return policy, unparsed_args

    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

        # extract and scale feature matrix
        state_feature_matrix = np.array([
            self.feature_extractor.extract(s)
            for s in self.update_batch_s
        ])
        # state_feature_matrix = self.feature_scaler.scale_features(state_feature_matrix, True)

        # add intercept
        intercept_state_feature_matrix = np.ones(shape=np.add(state_feature_matrix.shape, (0, 1)))
        intercept_state_feature_matrix[:, 1:] = state_feature_matrix

        action_matrix = np.array([
            a.value
            for a in self.update_batch_a
        ])

        # calculate per-update gradients
        gradients_a_s_mean, gradients_a_s_cov = self.get_action_density_gradients_vmap(
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
            gradients_a_s_mean,
            gradients_a_s_cov
        )

        for a, s, state_features, alpha, discounted_return, gradient_a_s_mean, gradient_a_s_cov in updates:

            # TODO:  This is always 1. How to estimate it from the PDF? The multivariate_normal class has a CDF.
            p_a_s = self[s][a]

            # check for nans in the gradients and skip the update if any are found
            if np.isnan(gradient_a_s_mean).any() or np.isnan(gradient_a_s_cov).any():
                warnings.warn('Gradients contain np.nan value(s). Skipping update.')
            else:

                # check whether the covariance matrix resulting from the updated parameters will be be positive
                # definite, as the multivariate normal distribution requires this. assign the update only if it is so.
                new_theta_cov = self.theta_cov + alpha * discounted_return * (gradient_a_s_cov / p_a_s)
                cov = self.get_covariance_matrix(
                    new_theta_cov,
                    state_features
                )

                if is_positive_definite(cov):
                    self.theta_mean += alpha * discounted_return * (gradient_a_s_mean / p_a_s)
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
            feature_extractor: StateFeatureExtractor
    ):
        """
        Initialize the parameterized policy.

        :param feature_extractor: Feature extractor.
        """

        super().__init__()

        self.feature_extractor = feature_extractor

        self.state_space_dimensionality = self.feature_extractor.get_state_space_dimensionality()
        self.action_space_dimensionality = self.feature_extractor.get_action_space_dimensionality()

        # coefficients for multi-dimensional mean:  one row per action and one column per state feature (plus 1 for the
        # bias/intercept).
        self.theta_mean = np.zeros(shape=(self.action_space_dimensionality, self.state_space_dimensionality + 1))

        # coefficients for multi-dimensional covariance:  one row per entry in the covariance matrix (a square matrix
        # with each action along each dimension) and one column per state feature (plus 1 for the bias/intercept).
        self.theta_cov = np.zeros(shape=(self.action_space_dimensionality ** 2, self.state_space_dimensionality + 1))

        self.get_action_density_gradients = jit(grad(self.get_action_density, argnums=(0, 1)))
        self.get_action_density_gradients_vmap = jit(vmap(self.get_action_density_gradients, in_axes=(None, None, 0, 0)))
        self.random_state = RandomState(12345)
        self.feature_scaler = NonstationaryFeatureScaler(
            1000,
            50000,
            0.99999
        )

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

    def __getitem__(
            self,
            state: MdpState
    ) -> Dict[Action, float]:
        """
        Get action-probability dictionary for a state.

        :param state: State.
        :return: Dictionary of action-probability items.
        """

        state_features = self.feature_extractor.extract(state)
        # state_features = self.feature_scaler.scale_features(np.array([state_features]), True)[0, :]
        state_features = np.append([1.0], state_features)

        # calculate the modeled mean and covariance of the n-dimensional action
        mean = self.theta_mean.dot(state_features)
        cov = self.get_covariance_matrix(
            self.theta_cov,
            state_features
        )

        # sample the n-dimensional action
        action_value = stats.multivariate_normal.rvs(mean=mean, cov=cov, random_state=self.random_state)

        # convert scalar action to array (e.g., if we're dealing with a 1-dimensional action)
        if np.isscalar(action_value):
            action_value = np.array([action_value])

        # sample action
        a = ContinuousMultiDimensionalAction(
            value=action_value,
            min_values=None,
            max_values=None
        )

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
