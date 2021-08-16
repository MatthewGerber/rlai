from argparse import ArgumentParser
from functools import reduce
from typing import List, Tuple, Dict

import numpy as np
from jax import numpy as jnp, grad

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.meta import rl_text
from rlai.policies import Policy
from rlai.policies.parameterized import ParameterizedPolicy
from rlai.q_S_A.function_approximation.models import FeatureExtractor
from rlai.states.mdp import MdpState
from rlai.utils import parse_arguments, load_class


@rl_text(chapter=13, page=322)
class SoftMaxInActionPreferencesPolicy(ParameterizedPolicy):
    """
    Parameterized policy that implements a soft-max over action preferences. The policy gradient calculation is coded up
    manually. See the `SoftMaxInActionPreferencesJaxPolicy` for a similar policy in which the gradient is calculated
    using the JAX library. This is only compatible with feature extractors derived from
    `rlai.q_S_A.function_approximation.models.feature_extraction.FeatureExtractor`, which return state-action feature
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
        if len(vars(parsed_args)) > 0:  # pragma no cover
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
            self.update_batch_target
        )

        for a, s, alpha, target in updates:
            gradient_a_s = self.gradient(a, s)
            p_a_s = self[s][a]
            self.theta += alpha * target * (gradient_a_s / p_a_s)

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

        if state is None:  # pragma no cover
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

    def __eq__(
            self,
            other
    ) -> bool:
        """
        Check whether the current policy equals another.

        :param other: Other policy.
        :return: True if policies are equal and False otherwise.
        """

        other: SoftMaxInActionPreferencesPolicy

        return np.allclose(self.theta, other.theta)

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


@rl_text(chapter=13, page=322)
class SoftMaxInActionPreferencesJaxPolicy(ParameterizedPolicy):
    """
    Parameterized policy that implements a soft-max over action preferences. The policy gradient calculation is
    performed using the JAX library. This is only compatible with feature extractors derived from
    `rlai.q_S_A.function_approximation.models.feature_extraction.FeatureExtractor`, which return state-action feature
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
        if len(vars(parsed_args)) > 0:  # pragma no cover
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
            self.update_batch_target
        )

        for a, s, alpha, target in updates:
            gradient_a_s = self.gradient(a, s)
            p_a_s = self[s][a]
            self.theta += alpha * target * (gradient_a_s / p_a_s)

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

        if state is None:  # pragma no cover
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

        :return: State dictionary.
        """

        state = dict(self.__dict__)

        state['get_action_prob_gradient'] = None

        return state

    def __setstate__(
            self,
            state: Dict
    ):
        """
        Set unpickled state.

        :param state: Unpickled state.
        """

        state['get_action_prob_gradient'] = grad(self.get_action_prob)

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

        other: SoftMaxInActionPreferencesJaxPolicy

        return np.allclose(self.theta, other.theta)

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
