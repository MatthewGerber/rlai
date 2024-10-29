from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict

import numpy as np
from numpy.random import RandomState

from rlai.core import Reward, Action, MdpState, Environment
from rlai.core.environments.mdp import ModelBasedMdpEnvironment, MdpEnvironment
from rlai.docs import rl_text
from rlai.gpi.state_action_value.function_approximation.models.feature_extraction import (
    StateActionInteractionFeatureExtractor
)
from rlai.models.feature_extraction import FeatureExtractor
from rlai.state_value.function_approximation.models.feature_extraction import StateFeatureExtractor
from rlai.utils import parse_arguments


@rl_text(chapter=3, page=60)
class Gridworld(ModelBasedMdpEnvironment):
    """
    Gridworld MDP environment.
    """

    @staticmethod
    def example_4_1(
            random_state: RandomState,
            T: Optional[int]
    ) -> 'Gridworld':
        """
        Construct the Gridworld for Example 4.1.

        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.

        :return: Gridworld.
        """

        RR = [
            Reward(
                i=i,
                r=r
            )
            for i, r in enumerate([0, -1])
        ]

        r_zero, r_minus_one = RR

        g = Gridworld(
            name='Example 4.1',
            random_state=random_state,
            T=T,
            n_rows=4,
            n_columns=4,
            terminal_states=[(0, 0), (3, 3)],
            RR=RR
        )

        # set nonterminal reward probabilities
        for a in [g.a_up, g.a_down, g.a_left, g.a_right]:

            # arrange grid such that a row-to-row scan will generate the appropriate state transition sequences for the
            # current action.
            if a == g.a_down:
                grid = g.grid
            elif a == g.a_up:
                grid = np.flipud(g.grid)
            elif a == g.a_right:
                grid = g.grid.transpose()
            elif a == g.a_left:
                grid = np.flipud(g.grid.transpose())
            else:  # pragma no cover
                raise ValueError(f'Unknown action:  {a}')

            # go row by row, with the final row transitioning to itself
            for s_row_i, s_prime_row_i in zip(range(grid.shape[0]), list(range(1, grid.shape[0])) + [-1]):
                for s, s_prime in zip(grid[s_row_i, :], grid[s_prime_row_i, :]):
                    if not s.terminal:
                        g.p_S_prime_R_given_S_A[s][a][s_prime][r_minus_one] = 1.0

        # set terminal reward probabilities
        for s in g.SS:
            if s.terminal:
                for a in s.AA:
                    g.p_S_prime_R_given_S_A[s][a][s][r_zero] = 1.0

        g.check_marginal_probabilities()

        return g

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
            '--id',
            type=str,
            default='example_4_1',
            help='Gridworld identifier.',
            choices=['example_4_1']
        )

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[Environment, List[str]]:
        """
        Initialize an environment from arguments.

        :param args: Arguments.
        :param random_state: Random state.
        :return: 2-tuple of an environment and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = parse_arguments(cls, args)

        gridworld_id = parsed_args.id
        del parsed_args.id

        gridworld = getattr(cls, gridworld_id)(
            random_state=random_state,
            **vars(parsed_args)
        )

        return gridworld, unparsed_args

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
            n_rows: int,
            n_columns: int,
            terminal_states: List[Tuple[int, int]],
            RR: List[Reward]
    ):
        """
        Initialize the gridworld.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        :param n_rows: Number of row.
        :param n_columns: Number of columns.
        :param terminal_states: List of terminal-state locations.
        :param RR: List of all possible rewards.
        """

        AA = [
            Action(
                i=i,
                name=direction
            )
            for i, direction in enumerate(['u', 'd', 'l', 'r'])
        ]

        self.a_up, self.a_down, self.a_left, self.a_right = AA

        SS = [
            MdpState(
                i=row_i * n_columns + col_j,
                AA=AA,
                terminal=False,
                truncated=False
            )
            for row_i in range(n_rows)
            for col_j in range(n_columns)
        ]

        for row, col in terminal_states:
            SS[row * n_columns + col].terminal = True

        super().__init__(
            name=name,
            random_state=random_state,
            T=T,
            SS=SS,
            RR=RR
        )

        self.grid = np.array(self.SS).reshape(n_rows, n_columns)


@rl_text(chapter='Feature Extractors', page=1)
class GridworldFeatureExtractor(StateActionInteractionFeatureExtractor):
    """
    A feature extractor for the gridworld. This extractor, being based on the `StateActionInteractionFeatureExtractor`,
    directly extracts the fully interacted state-action feature matrix. It returns numpy.ndarray feature matrices, which
    are not compatible with the Patsy formula-based interface.
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

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        assert isinstance(environment, Gridworld)

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls(
            environment=environment
        )

        return fex, unparsed_args

    def extracts_intercept(
            self
    ) -> bool:
        """
        Whether the feature extractor extracts an intercept (constant) term.

        :return: True if an intercept (constant) term is extracted and False otherwise.
        """

        return True

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract features for state-action pairs.

        :param states: States.
        :param actions: Actions.
        :param refit_scaler: Whether to refit the feature scaler before scaling the extracted features. This is
        only appropriate in settings where nonstationarity is desired (e.g., during training). During evaluation, the
        scaler should remain fixed, which means this should be False.
        :return: State-feature numpy.ndarray.
        """

        self.check_state_and_action_lists(states, actions)

        state_row = [int(state.i / self.num_cols) for state in states]  # type: ignore[operator]
        state_col = [state.i % self.num_cols for state in states]  # type: ignore[operator]

        state_features = np.array([
            [
                1.0,  # intercept
                row,  # from top
                self.num_rows - row - 1,  # from bottom
                col,  # from left
                self.num_cols - col - 1  # from right
            ]
            for row, col in zip(state_row, state_col)
        ])

        return self.interact(
            state_features=state_features,
            actions=actions,
            refit_scaler=refit_scaler
        )

    def get_action_feature_names(
            self
    ) -> Optional[Dict[str, List[str]]]:
        """
        Get names of actions and their associated feature names.

        :return: Dictionary of action names and their associated feature names.
        """

        return {
            a.name: ['intercept', 'from-top', 'from-bottom', 'from-left', 'from-right']  # type: ignore[misc]
            for a in self.actions
        }

    def __init__(
            self,
            environment: Gridworld
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        # don't rescale features in the superclass action-interactor. no rescaling at all in this class.
        super().__init__(
            environment=environment,
            actions=[
                environment.a_up,
                environment.a_down,
                environment.a_left,
                environment.a_right
            ],
            scale_features=False
        )

        self.num_rows = environment.grid.shape[0]
        self.num_cols = environment.grid.shape[1]


@rl_text(chapter='Feature Extractors', page=1)
class GridworldStateFeatureExtractor(StateFeatureExtractor):
    """
    A feature extractor for the gridworld. This extractor does not interact feature values with actions. Its primary use
    is in state-value estimation (e.g., for the baseline of policy gradient methods).
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

        return parser

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :param environment: Environment.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        assert isinstance(environment, Gridworld)

        parsed_args, unparsed_args = parse_arguments(cls, args)

        # there shouldn't be anything left
        if len(vars(parsed_args)) > 0:  # pragma no cover
            raise ValueError('Parsed args remain. Need to pass to constructor.')

        fex = cls(
            environment=environment
        )

        return fex, unparsed_args

    def extracts_intercept(
            self
    ) -> bool:
        """
        Whether the feature extractor extracts an intercept (constant) term.

        :return: True if an intercept (constant) term is extracted and False otherwise.
        """

        return False

    def extract(
            self,
            states: List[MdpState],
            refit_scaler: bool
    ) -> np.ndarray:
        """
        Extract state features.

        :param states: States.
        :param refit_scaler: Unused.
        :return: State-feature matrix (#states, #features).
        """

        state_indices = [
            state.i
            for state in states
            if state.i is not None
        ]

        row_col_list = [
            (
                i / self.num_cols,
                i % self.num_cols
            )
            for i in state_indices
        ]

        state_feature_matrix = np.array([
            [
                row,  # from top
                self.num_rows - row - 1,  # from bottom
                col,  # from left
                self.num_cols - col - 1  # from right
            ]
            for row, col in row_col_list
        ])

        return state_feature_matrix

    def get_feature_names(
            self
    ) -> Optional[List[str]]:
        """
        Get names of features.

        :return: List of feature names.
        """

        return ['from-top', 'from-bottom', 'from-left', 'from-right']

    def __init__(
            self,
            environment: Gridworld
    ):
        """
        Initialize the feature extractor.

        :param environment: Environment.
        """

        super().__init__(False)

        self.num_rows = environment.grid.shape[0]
        self.num_cols = environment.grid.shape[1]
