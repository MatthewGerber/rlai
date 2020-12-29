import sys
from argparse import ArgumentParser
from typing import List

from rlai.meta import rl_text
from rlai.runners import trainer, agent_in_environment
from rlai.utils import load_class, parse_arguments


@rl_text(chapter='Training and Running Agents', page=1)
def run(
        args: List[str] = None
):
    """
    Run RLAI.

    :param args: Arguments.
    """

    # create the top-level rlai parser and add subparsers for commands
    parser = ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers()

    # train
    train_parser = subparsers.add_parser('train', add_help=False)
    train_parser.set_defaults(func=trainer.run)

    # run in environment
    run_parser = subparsers.add_parser('run', add_help=False)
    run_parser.set_defaults(func=agent_in_environment.run)

    # help
    help_parser = subparsers.add_parser('help', add_help=False)
    help_parser.set_defaults(func=show_help)

    parsed_args, unparsed_args = parser.parse_known_args(args)
    parsed_args.func(unparsed_args)


def show_help(
        args: List[str]
):
    """
    Show help for a class.

    :param args: Arguments.
    """

    if len(args) != 1:
        print('Usage:  rlai help CLASS')

    try:
        loaded_class = load_class(args[0])
        parser = loaded_class.get_argument_parser()
        parse_arguments(parser, ['--help'])
    except Exception as ex:
        print(f'{ex}')


if __name__ == '__main__':
    run(sys.argv[1:])
