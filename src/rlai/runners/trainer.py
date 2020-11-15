import sys
from argparse import ArgumentParser
from typing import List

from numpy.random import RandomState

from rlai.gpi.utils import resume_from_checkpoint
from rlai.utils import import_function, load_class


def run(
        args: List[str]
):
    parser = ArgumentParser(description='Run the trainer.')

    parser.add_argument(
        '--train',
        action='store_true',
        help='Train an agent in an environment.'
    )

    parser.add_argument(
        '--agent',
        type=str,
        help='Fully-qualified type name of agent to train.'
    )

    parser.add_argument(
        '--environment',
        type=str,
        help='Fully-qualified type name of environment to train agent in.'
    )

    parser.add_argument(
        '--train-function',
        type=str,
        help='Fully-qualified type name of function to use for training the agent.'
    )

    parser.add_argument(
        '--resume-train',
        action='store_true',
        help='Resume training an agent from a checkpoint path.'
    )

    parsed_args, unparsed_args = parser.parse_known_args(args)

    train_function = import_function(parsed_args.train_function)

    random_state = RandomState(12345)

    if parsed_args.train:

        agent_class = load_class(parsed_args.agent)
        agents, unparsed_args = agent_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

        if len(agents) != 1:
            raise ValueError('Training is only supported for single agents. Please specify one agent.')

        agent = agents[0]

        environment_class = load_class(parsed_args.environment)
        environment, unparsed_args = environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

        function_args_parser = ArgumentParser('Training function argument parser')

        function_args_parser.add_argument(
            '--num-improvements',
            type=int,
            help='Number of improvements.'
        )

        function_args_parser.add_argument(
            '--num-episodes-per-improvement',
            type=int,
            help='Number of episodes per improvement.'
        )

        function_args_parser.add_argument(
            '--update-upon-every-visit',
            type=bool,
            help='Whether or not to update values upon each visit.'
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

        function_args_parser.add_argument(
            '--mode',
            type=str,
            help='Q-learning mode (SARSA, Q_LEARNING, EXPECTED_SARSA).'
        )

        function_args_parser.add_argument(
            '--n-steps',
            type=int,
            help='N-step update value.'
        )

        parsed_train_function_args, unparsed_train_function_args = function_args_parser.parse_known_args(unparsed_args)

        if len(unparsed_train_function_args) > 0:
            raise ValueError(f'Unparsed training function arguments remain:  {unparsed_train_function_args}')

        train_function(
            agent=agent,
            environment=environment,
            **{
                arg: v
                for arg, v in vars(parsed_train_function_args).items()
                if v is not None
            }
        )

    elif parsed_args.resume_train:

        # some environments cannot be pickled (e.g., gym). provide a default if we lack one.
        if hasattr(parsed_args, 'environment'):
            environment_class = load_class(parsed_args.environment)
            default_environment, _ = environment_class.init_from_arguments(
                args=unparsed_args,
                random_state=random_state
            )
        else:
            default_environment = None

        train_function = import_function(parsed_args.train_function)

        resume_from_checkpoint(
            checkpoint_path=parsed_args.train_checkpoint_path,
            resume_function=train_function,
            default_environment=default_environment
        )


if __name__ == '__main__':
    run(sys.argv[1:])
