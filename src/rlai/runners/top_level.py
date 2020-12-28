import sys
from argparse import ArgumentParser
from typing import List

from rlai.meta import rl_text
from rlai.runners import trainer, agent_in_environment


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

    parsed_args, unparsed_args = parser.parse_known_args(args)
    parsed_args.func(unparsed_args)


if __name__ == '__main__':
    run(sys.argv[1:])
