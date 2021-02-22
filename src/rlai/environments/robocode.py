from typing import List, Union, Tuple, Dict, Any, Optional

import numpy as np
from numpy.random import RandomState

from rlai.actions import Action
from rlai.environments.mdp import MdpEnvironment
from rlai.environments.rest import RestMdpEnvironment
from rlai.meta import rl_text
from rlai.states.mdp import MdpState
from rlai.value_estimation.function_approximation.models import StateActionInteractionFeatureExtractor


@rl_text(chapter='Environments', page=1)
class RobocodeEnvironment(RestMdpEnvironment):
    """
    Robocode environment.
    """

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            random_state: RandomState
    ) -> Tuple[MdpEnvironment, List[str]]:

        return RobocodeEnvironment(
            name='test',
            random_state=random_state,
            T=None
        ), args

    def init_state_from_rest_request_dict(
            self,
            rest_request_dict: Dict[Any, Any],
            terminal: bool
    ) -> MdpState:

        return RobocodeState(
            **rest_request_dict,
            actions=self.robot_actions,
            terminal=terminal
        )

    def __init__(
            self,
            name: str,
            random_state: RandomState,
            T: Optional[int],
    ):
        """
        Initialize the MDP environment.

        :param name: Name.
        :param random_state: Random state.
        :param T: Maximum number of steps to run, or None for no limit.
        """

        super().__init__(
            name=name,
            random_state=random_state,
            T=T
        )

        self.robot_actions = [
            Action(0, 'ahead'),
            Action(1, 'back'),
            Action(2, 'fire')
        ]


class RobocodeState(MdpState):

    def __init__(
            self,
            x: float,
            y: float,
            scanned_robot_event: Dict[Any, Any],
            dead: bool,
            win: bool,
            actions: List[Action],
            terminal: bool
    ):
        super().__init__(
            i=None,
            AA=actions,
            terminal=terminal
        )

        self.x = x
        self.y = y
        self.scanned_robot_event = scanned_robot_event
        self.dead = dead
        self.win = win


class RobocodeFeatureExtractor(StateActionInteractionFeatureExtractor):

    @classmethod
    def init_from_arguments(
            cls,
            args: List[str],
            environment: MdpEnvironment
    ) -> Tuple[StateActionInteractionFeatureExtractor, List[str]]:

        return RobocodeFeatureExtractor(environment, [Action(1), Action(2), Action(3)]), args

    def extract(
            self,
            states: List[MdpState],
            actions: List[Action],
            for_fitting: bool
    ) -> Union[np.ndarray]:

        return np.array([
            [1, 2, 3]
            for a in actions
        ])
