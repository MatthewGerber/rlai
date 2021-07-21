import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from functools import reduce
from typing import Dict, List, Tuple, Set

import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
from jax import grad
from numpy.random import RandomState
from scipy import stats

from rlai.actions import Action, ContinuousMultiDimensionalAction
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
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

    @abstractmethod
    def update(
            self,
            a: Action,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ):
        """
        Update the policy parameters for an action in a state using a discounted return and a step size.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        """


@rl_text(chapter=13, page=322)
class SoftMaxInActionPreferencesPolicy(ParameterizedPolicy):
    """
    Parameterized policy that implements a soft-max over action preferences. The policy gradient calculation is coded up
    manually. See the `JaxSoftMaxInActionPreferencesPolicy` for a similar policy in which the gradient is calculated
    using the JAX library.
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

    def update(
            self,
            a: Action,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ):
        """
        Update the policy parameters for an action in a state using a discounted return and a step size.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        """

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
    performed using the JAX library.
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

    def update(
            self,
            a: Action,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ):
        """
        Get the policy parameter update for an action in a state using a discounted return and a step size.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        """

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
    distribution (e.g., the multidimensional mean and covariance matrixf) in terms of state features.
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

    def update(
            self,
            a: ContinuousMultiDimensionalAction,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ):
        """
        Update the policy parameters for an action in a state using a discounted return and a step size.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        """

        # TODO:  This is always 1. How to estimate it from the PDF? The multivariate_normal class has a CDF.
        p_a_s = self[s][a]

        gradient_a_s_mean, gradient_a_s_cov = self.gradient(
            self.theta_mean,
            self.theta_cov,
            self.theta_cov_diagonal_rows,
            a,
            s
        )

        # check for nans in the gradient and skip update if any are found
        if np.isnan(gradient_a_s_mean).any() or np.isnan(gradient_a_s_cov).any():
            warnings.warn('Gradients are NaN. Skipping update.')
        else:

            self.theta_mean += alpha * discounted_return * (gradient_a_s_mean / p_a_s)

            # check whether the covariance matrix resulting from the updated parameters will be be positive definite, as
            # it must be for multivariate normal distribution. don't update theta if it won't be.
            new_theta_cov = self.theta_cov + alpha * discounted_return * (gradient_a_s_cov / p_a_s)
            state_features = self.feature_extractor.extract(s)
            covariance = self.get_covariance_matrix(new_theta_cov, state_features, self.theta_cov_diagonal_rows, len(self.theta_mean))
            if is_positive_definite(covariance):
                self.theta_cov = new_theta_cov
            else:
                warnings.warn('The updated covariance theta parameters produce a covariance matrix that is not positive definite. Skipping update.')

    @staticmethod
    def get_covariance_matrix(
            theta_cov: np.ndarray,
            state_features: np.ndarray,
            theta_cov_diagonal_rows: Set[int],
            action_dimensionality: int
    ) -> np.ndarray:
        """
        Get covariance matrix from its parameters.

        :param theta_cov: Parameters.
        :param state_features: State features.
        :param theta_cov_diagonal_rows: Rows that correspond to the diagonal in the covariance matrix.
        :param action_dimensionality: Dimensionality of action space.
        :return: Covariance matrix.
        """

        # ensure that the diagonal of the covariance matrix has positive values by exponentiating
        return np.array([
            np.exp(np.dot(row, state_features)) if i in theta_cov_diagonal_rows else np.dot(row, state_features)
            for i, row in enumerate(theta_cov)
        ]).reshape(action_dimensionality, action_dimensionality)

    def gradient(
            self,
            theta_mean: np.ndarray,
            theta_cov: np.ndarray,
            theta_cov_diagonal_rows: Set[int],
            a: ContinuousMultiDimensionalAction,
            s: MdpState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the gradient of the policy for an action in a state, with respect to the policy's parameters.

        :param theta_mean: Policy parameters for mean.
        :param theta_cov: Policy parameters for covariance matrix.
        :param theta_cov_diagonal_rows: Rows in `theta_cov` that correspond to diagonal elements of the covariance
        matrix.
        :param a: Action.
        :param s: State.
        :return: 2-tuple of partial gradient vectors, one vector for the mean and one vector for the covariance matrix.
        """

        state_features = self.feature_extractor.extract(s)

        gradient_mean, gradient_std = self.get_action_density_gradients(
            theta_mean,
            theta_cov,
            theta_cov_diagonal_rows,
            state_features,
            a.value
        )

        return gradient_mean, gradient_std

    @staticmethod
    def get_action_density(
            theta_mean: np.ndarray,
            theta_cov: np.ndarray,
            theta_cov_diagonal_rows: Set[int],
            state_features: np.ndarray,
            action_value: np.ndarray
    ) -> float:
        """
        Get action density.

        :param theta_mean: Policy parameters for mean.
        :param theta_cov: Policy parameters for covariance matrix.
        :param theta_cov_diagonal_rows: Rows in `theta_cov` that correspond to diagonal elements of the covariance
        matrix.
        :param state_features: A vector of state features.
        :param action_value: Action value.
        :return: Action density.
        """

        mean = jnp.dot(theta_mean, state_features)
        action_dimensionality = len(mean)
        cov = jnp.array([
            jnp.exp(jnp.dot(row, state_features)) if i in theta_cov_diagonal_rows else jnp.dot(row, state_features)
            for i, row in enumerate(theta_cov)
        ]).reshape(action_dimensionality, action_dimensionality)

        return jstats.multivariate_normal.pdf(action_value, mean, cov)

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

        state_space_dimensionality = self.feature_extractor.get_state_space_dimensionality()
        action_space_dimensionality = self.feature_extractor.get_action_space_dimensionality()

        # coefficients for multi-dimensional mean:  one row per action and one column per feature
        self.theta_mean = np.zeros(shape=(action_space_dimensionality, state_space_dimensionality))

        # coefficients for multi-dimensional covariance:  one row per entry in the covariance matrix and one column per
        # feature. start with a diagonal covariance matrix and flatten it.
        diagonal_cov = np.zeros(shape=(action_space_dimensionality, action_space_dimensionality))
        np.fill_diagonal(diagonal_cov, 1.0)
        self.theta_cov = np.array([
            np.ones(state_space_dimensionality) if cov == 1.0 else np.zeros(state_space_dimensionality)
            for cov_row in diagonal_cov
            for cov in cov_row
        ])

        # keep track of which rows correspond to the diagonal, since we need to ensure that the resulting covariance
        # matrix has positive entries along the diagonal (we'll do something like exponentiate the dot-product to
        # ensure this).
        self.theta_cov_diagonal_rows = set(
            i * action_space_dimensionality + j
            for i, j in enumerate(range(action_space_dimensionality))
        )

        self.get_action_density_gradients = grad(self.get_action_density, argnums=(0, 1))
        self.rng = RandomState(12345)

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

        # calculate the modeled mean and covariance of the n-dimensional action
        mean = self.theta_mean.dot(state_features)
        covariance = self.get_covariance_matrix(
            self.theta_cov,
            state_features,
            self.theta_cov_diagonal_rows,
            len(mean)
        )

        # sample action
        a = ContinuousMultiDimensionalAction(
            value=stats.multivariate_normal.rvs(mean=mean, cov=covariance, random_state=self.rng),
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

        return state_dict

    def __setstate__(
            self,
            state_dict: Dict
    ):
        """
        Set unpickled state.

        :param state_dict: Unpickled state.
        """

        state_dict['get_action_density_gradients'] = grad(self.get_action_density, argnums=(0, 1))

        self.__dict__ = state_dict
