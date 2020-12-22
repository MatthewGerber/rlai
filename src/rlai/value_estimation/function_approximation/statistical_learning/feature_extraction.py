from abc import ABC, abstractmethod
from typing import List

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
    Simple state-action identity extractor.
    """

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
