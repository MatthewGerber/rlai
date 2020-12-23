from abc import ABC, abstractmethod
from argparse import Namespace, ArgumentParser
from typing import List, Tuple

import pandas as pd

from rlai.actions import Action
from rlai.meta import rl_text
from rlai.states.mdp import MdpState


@rl_text(chapter=9, page=197)
class FeatureExtractor(ABC):
    """
    Feature extractor.
    """

    @abstractmethod
    def extract(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> pd.DataFrame:
        """
        Extract features from a state and actions.

        :param state: State.
        :param actions: Actions.
        :return: DataFrame with one row per action and one column per feature.
        """
        pass


class StateActionIdentityFeatureExtractor(FeatureExtractor):
    """
    Simple state/action identifier extractor. Generates features named "s" and "a" for each observation. The
    interpretation of the feature values (i.e., state and action identifiers) depends on the environment. The values
    are always integers, but whether they are ordinal (ordered) or categorical (unordered) depends on the environment.
    Furthermore, it should not be assumed that the environment will provide such identifiers. They will generally be
    provided for actions (which are generally easy to enumerate up front), but this is certainly not true for states,
    which are not (easily) enumerable for all environments. All of this to say that this feature extractor is not
    generally useful. You should consider writing your own feature extractor for your environment.
    """

    @classmethod
    def parse_arguments(
            cls,
            args
    ) -> Tuple[Namespace, List[str]]:
        """
        Parse arguments.

        :param args: Arguments.
        :return: 2-tuple of parsed and unparsed arguments.
        """

        parser = ArgumentParser(allow_abbrev=False)

        # future arguments to be added here...

        parsed_args, unparsed_args = parser.parse_known_args(args)

        return parsed_args, unparsed_args

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str]
    ) -> Tuple[FeatureExtractor, List[str]]:
        """
        Initialize a feature extractor from arguments.

        :param args: Arguments.
        :return: 2-tuple of a feature extractor and a list of unparsed arguments.
        """

        parsed_args, unparsed_args = cls.parse_arguments(args)

        fex = StateActionIdentityFeatureExtractor()

        return fex, unparsed_args

    def extract(
            self,
            state: MdpState,
            actions: List[Action]
    ) -> pd.DataFrame:
        """
        Extract discrete state and action identifiers.

        :param state: State.
        :param actions: Actions.
        :return: DataFrame with one row per action and two columns (one for the state and one for the action).
        """

        return pd.DataFrame([
            (state.i, action.i)
            for action in actions
        ], columns=['s', 'a'])
