from abc import ABC, abstractmethod
from argparse import ArgumentParser

from rlai.core import Policy, Action, MdpState
from rlai.docs import rl_text
from rlai.utils import get_base_argument_parser


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
            target: float
    ):
        """
        Append an update for an action in a state using a target and a step size. All appended updates will be committed
        to the policy when `commit_updates` is called.

        :param a: Action.
        :param s: State.
        :param alpha: Step size.
        :param target: Update target.
        """

        self.update_batch_a.append(a)
        self.update_batch_s.append(s)
        self.update_batch_alpha.append(alpha)
        self.update_batch_target.append(target)
        self.updates_available = True

    def commit_updates(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`.
        """

        if self.updates_available:
            self.__commit_updates__()
            self.update_batch_a.clear()
            self.update_batch_s.clear()
            self.update_batch_alpha.clear()
            self.update_batch_target.clear()
            self.updates_available = False

    @abstractmethod
    def __commit_updates__(
            self
    ):
        """
        Commit updates that were previously appended with calls to `append_update`. Not intended to be called directly
        by outside callers or inheritors.
        """

    def close(
            self
    ):
        """
        Close the policy, releasing any resources that it holds (e.g., display windows for plotting).
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
        self.update_batch_target = []
        self.updates_available = False
