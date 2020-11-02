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
        '--resume-train',
        action='store_true',
        help='Resume training a Mancala agent from the checkpoint path.'
    )

    parser.add_argument(
        '--play-human',
        action='store_true',
        help='Let a human play the trained agent interactively.'
    )

    parsed_args, unparsed_args = parser.parse_known_args(args)

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

        function_args_parser = ArgumentParser('Training function argument parser')

        function_args_parser.add_argument(
            '--num-improvements',
            type=int,
            default=1000,
            help='Number of improvements.'
        )

        function_args_parser.add_argument(
            '--num-episodes-per-improvement',
            type=int,
            default=500,
            help='Number of episodes per improvement.'
        )

        function_args_parser.add_argument(
            '--update-upon-every-visit',
            type=bool,
            default=False,
            help='Whether or not to update Q-values upon each visit.'
        )

        function_args_parser.add_argument(
            '--alpha',
            type=float,
            help='Step size.'
        )

        function_args_parser.add_argument(
            '--epsilon',
            type=float,
            help='Total probability mass to allocate across all policy actions.'
        )

        function_args_parser.add_argument(
            '--num-improvements-per-plot',
            type=int,
            help='Number of improvements per plot.'
        )

        function_args_parser.add_argument(
            '--num-improvements-per-checkpoint',
            type=int,
            help='Number of improvements per checkpoint.'
        )

        function_args_parser.add_argument(
            '--checkpoint-path',
            type=str,
            help='Path to checkpoint file.'
        )

        parsed_train_function_args, unparsed_train_function_args = function_args_parser.parse_known_args(unparsed_args)

        if len(unparsed_train_function_args) > 0:
            raise ValueError(f'Unparsed training function arguments remain:  {unparsed_train_function_args}')

        train_function(
            agent=p1,
            environment=mancala,
            **dict((arg, v) for arg, v in parsed_train_function_args._get_kwargs() if v is not None)
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
