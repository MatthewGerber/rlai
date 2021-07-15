from abc import ABC
from argparse import ArgumentParser
from functools import reduce
from typing import Dict, List, Tuple, Any

import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
from jax import grad, jit, random as jrandom

from rlai.actions import Action, ContinuousAction
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.q_S_A.function_approximation.models import FeatureExtractor
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments, load_class, get_base_argument_parser


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

    def get_update(
            self,
            a: Action,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ) -> np.ndarray:
        """
        Get the policy parameter update for an action in a state using a discounted return and a step size.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        :return: Policy parameter (theta) update.
        """

        gradient_a_s = self.gradient(a, s)
        p_a_s = self[s][a]

        return alpha * discounted_return * (gradient_a_s / p_a_s)

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
    ) -> jnp.ndarray:
        """
        Get a matrix containing a feature vector for each action in a state.

        :param state: State.
        :return: An (#features, #actions) matrix.
        """

        return jnp.array([
            self.feature_extractor.extract([state], [a], False)[0, :]
            for a in state.AA
        ]).transpose()

    @staticmethod
    def get_action_prob(
            theta: jnp.ndarray,
            state_action_features: jnp.ndarray,
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

    def get_update(
            self,
            a: Action,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ) -> np.ndarray:
        """
        Get the policy parameter update for an action in a state using a discounted return and a step size.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        :return: Policy parameter (theta) update.
        """

        gradient_a_s = self.gradient(a, s)
        p_a_s = self[s][a]

        return alpha * discounted_return * (gradient_a_s / p_a_s)

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

        self.theta = jnp.zeros(sum(len(features) for features in feature_extractor.get_action_feature_names().values()))
        self.get_action_prob_gradient = jit(grad(self.get_action_prob))

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

        state_dict['get_action_prob_gradient'] = jit(grad(self.get_action_prob))

        self.__dict__ = state_dict


@rl_text(chapter=13, page=335)
class ContinuousActionDistributionPolicy(ParameterizedPolicy):
    """
    Parameterized policy that produces continuous actions by modeling a continuous distribution in terms of
    state-action feature vectors.
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

        action_state_features = self.get_state_action_features(s)[:, a.i]
        gradient = self.get_action_prob_gradient(self.theta, action_state_features, a.value)

        return gradient

    def get_state_action_features(
            self,
            state: MdpState
    ) -> jnp.ndarray:
        """
        Get a matrix containing a feature vector for each action in a state.

        :param state: State.
        :return: An (#actions, #features) matrix.
        """

        return jnp.array([
            self.feature_extractor.extract([state], [a], False)[0, :]
            for a in state.AA
        ]).transpose()

    @staticmethod
    def get_action_prob(
            theta: jnp.ndarray,
            state_action_features: jnp.ndarray,
            action_value: float
    ) -> float:
        """
        Get action probability.

        :param theta: Policy parameters.
        :param state_action_features: A vector of state-action features as returned by `get_state_action_features`.
        :param action_value: Action value.
        :return: Action probability.
        """

        mean, std = ContinuousActionDistributionPolicy.get_means_and_stds(theta, state_action_features)

        action_prob_upper = jstats.norm.cdf(action_value + std / 10.0, loc=mean, scale=std)
        action_prob_lower = jstats.norm.cdf(action_value - std / 10.0, loc=mean, scale=std)
        action_prob = action_prob_upper - action_prob_lower

        return action_prob

    @staticmethod
    def get_means_and_stds(
            theta: jnp.ndarray,
            state_action_features: jnp.ndarray
    ) -> Tuple[Any, Any]:
        """
        Get means and standard deviations for a matrix of state-action feature vectors.

        :param theta: Policy parameters.
        :param state_action_features: An (#features, #actions) Matrix of state-action feature vectors.
        :return: 2-tuple of means and standard deviations, with one mean and one standard deviation for each column
        vector in `state_action_features`.
        """

        theta_split = int(theta.shape[0] / 2)

        theta_mean = theta[0:theta_split]
        means = jnp.dot(theta_mean, state_action_features)

        theta_std = theta[theta_split:]
        stds = jnp.exp(jnp.dot(theta_std, state_action_features))

        return means, stds

    def get_update(
            self,
            a: Action,
            s: MdpState,
            alpha: float,
            discounted_return: float
    ) -> np.ndarray:
        """
        Get the policy parameter update for an action in a state using a discounted return and a step size.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param discounted_return: Discounted return.
        :return: Policy parameter (theta) update.
        """

        gradient_a_s = self.gradient(a, s)
        p_a_s = self[s][a]

        return alpha * discounted_return * (gradient_a_s / p_a_s)

    def get_random_key(
            self
    ) -> np.uint32:
        """
        Get a new random key.

        :return: Random key.
        """

        random_key = self.jax_rand_key
        _, self.jax_rand_key = jrandom.split(self.jax_rand_key)

        return random_key

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

        self.theta = jnp.zeros(sum(len(features) for features in feature_extractor.get_action_feature_names().values()) * 2)
        self.get_action_prob_gradient = jit(grad(self.get_action_prob))
        self.jax_rand_key = jrandom.PRNGKey(12345)

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

        # extract state-action features for state (#features, #actions)
        state_action_features = self.get_state_action_features(state)

        # calculate the modeled mean and standard deviation for each action
        means, stds = self.get_means_and_stds(
            theta=self.theta,
            state_action_features=state_action_features
        )

        # sample actions
        sampled_actions = [
            ContinuousAction(
                i=i,
                value=std * jrandom.normal(self.get_random_key()) + mean,
                min_value=None,
                max_value=None
            )
            for i, (mean, std) in enumerate(zip(means, stds))
        ]

        # calculate action probabilities
        action_prob = {
            a: self.get_action_prob(self.theta, state_action_features[:, i], a.value)
            for i, a in enumerate(sampled_actions)
        }

        # rescale to form distribution
        total = sum(action_prob.values())
        action_prob = {
            a: p / total
            for a, p in action_prob.items()
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

        state_dict['get_action_prob_gradient'] = jit(grad(self.get_action_prob))

        self.__dict__ = state_dict
