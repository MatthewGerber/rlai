import sys
from argparse import ArgumentParser
from typing import List

from numpy.random import RandomState

from rlai.agents.mdp import Human, StochasticMdpAgent
from rlai.environments.mancala import Mancala
from rlai.gpi.utils import resume_from_checkpoint
from rlai.utils import import_function


def human_player_mutator(
        environment: Mancala,
        **kwargs
):
    """
    Change the Mancala environment to let a human play the trained agent.

    :param environment: Environment.
    :param kwargs: Unused args.
    """
    environment.player_2 = Human()


def run(
        args: List[str]
):
    parser = ArgumentParser(description='Run the Mancala game.')

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a Mancala agent.'
    )

    parser.add_argument(
        '--train-function',
        type=str,
        help='Fully-qualified type name of function to use for training the Mancala agent.'
    )

    parser.add_argument(
        '--train-checkpoint-path',
        type=str,
        help='Path to checkpoint file.'
    )

    parser.add_argument(
        '--resume-train',
        action='store_true',
        help='Resume training a Mancala agent from the checkpoint path.'
    )

    parser.add_argument(
        '--play-human',
        action='store_true',
        help='Let a human play the trained agent interactively.'
    )

    parsed_args = parser.parse_args(args)

    if parsed_args.train:

        random_state = RandomState(12345)

        p2 = StochasticMdpAgent(
            'player 2',
            random_state,
            1
        )

        mancala: Mancala = Mancala(
            initial_count=4,
            random_state=random_state,
            player_2=p2
        )

        p1 = StochasticMdpAgent(
            'player 1',
            random_state,
            1
        )

        train_function = import_function(parsed_args.train_function)

        train_function(
            agent=p1,
            environment=mancala,
            num_improvements=1000,
            num_episodes_per_improvement=500,
            alpha=0.1,
            epsilon=0.05,
            num_improvements_per_plot=20,
            num_improvements_per_checkpoint=100,
            checkpoint_path=parsed_args.train_checkpoint_path
        )

    elif parsed_args.resume_train:

        train_function = import_function(parsed_args.train_function)

        resume_from_checkpoint(
            checkpoint_path=parsed_args.train_checkpoint_path,
            resume_function=train_function,
            num_improvements=500
        )

    elif parsed_args.play_human:

        train_function = import_function(parsed_args.train_function)

        resume_from_checkpoint(
            checkpoint_path=parsed_args.train_checkpoint_path,
            resume_function=train_function,
            resume_args_mutator=human_player_mutator,
            num_improvements=500
        )


if __name__ == '__main__':
    run(sys.argv[1:])
