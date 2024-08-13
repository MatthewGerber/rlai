import sys
from argparse import ArgumentParser
from typing import List, Union, Tuple, Optional

from rlai.docs import rl_text
from rlai.runners import trainer, agent_in_environment
from rlai.utils import parse_arguments, get_argument_parser


@rl_text(chapter='Training and Running Agents', page=1)
def run(
        args: Optional[List[str]] = None
) -> Union[None, Tuple[Optional[str], str]]:
    """
    Run RLAI.

    :param args: Arguments.
    :return: Return value of specified function.
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
    return parsed_args.func(unparsed_args)


def show_help(
        args: List[str]
):
    """
    Show help for a class.

    :param args: Arguments.
    """

    if len(args) == 1:
        try:
            parser = get_argument_parser(args[0])
            parse_arguments(parser, ['--help'])
        except Exception as ex:
            print(f'{ex}')
    else:
        print('Usage:  rlai help CLASS')


if __name__ == '__main__':  # pragma no cover
    run(sys.argv[1:])
