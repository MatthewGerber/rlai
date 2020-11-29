import os
import pickle
import sys
import warnings
from argparse import ArgumentParser
from typing import List, Tuple, Optional

from numpy.random import RandomState

from rlai.gpi.utils import resume_from_checkpoint
from rlai.meta import rl_text
from rlai.utils import import_function, load_class


@rl_text(chapter='Training and Running Agents', page=1)
def run(
        args: List[str]
) -> Tuple[Optional[str], str]:
    """
    Train an agent in an environment.

    :param args: Arguments.
    :returns: 2-tuple of the checkpoint path (if any) and the saved agent path.
    """

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

    parser.add_argument(
        '--save-agent-path',
        type=str,
        help='Path to store resulting agent to.'
    )

    parsed_args, unparsed_args = parser.parse_known_args(args)

    if parsed_args.save_agent_path is None:
        raise ValueError('No --save-agent-path specified. Cannot save agent.')

    train_function_arg_parser = ArgumentParser('Training function argument parser')

    train_function_arg_parser.add_argument(
        '--num-improvements',
        type=int,
        help='Number of improvements.'
    )

    train_function_arg_parser.add_argument(
        '--num-episodes-per-improvement',
        type=int,
        help='Number of episodes per improvement.'
    )

    train_function_arg_parser.add_argument(
        '--update-upon-every-visit',
        type=str,
        choices=['True', 'False'],
        help='Whether or not to update values upon each visit to a state or state-action pair.'
    )

    train_function_arg_parser.add_argument(
        '--alpha',
        type=float,
        help='Step size.'
    )

    train_function_arg_parser.add_argument(
        '--epsilon',
        type=float,
        help='Total probability mass to allocate across all policy actions.'
    )

    train_function_arg_parser.add_argument(
        '--make-final-policy-greedy',
        type=str,
        choices=['True', 'False'],
        help='Whether or not to make the final policy greedy after training is complete.'
    )

    train_function_arg_parser.add_argument(
        '--num-improvements-per-plot',
        type=int,
        help='Number of improvements per plot.'
    )

    train_function_arg_parser.add_argument(
        '--num-improvements-per-checkpoint',
        type=int,
        help='Number of improvements per checkpoint.'
    )

    train_function_arg_parser.add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to checkpoint file.'
    )

    train_function_arg_parser.add_argument(
        '--mode',
        type=str,
        help='Temporal difference evaluation mode (SARSA, Q_LEARNING, EXPECTED_SARSA).'
    )

    train_function_arg_parser.add_argument(
        '--n-steps',
        type=int,
        help='N-step update value.'
    )

    train_function_arg_parser.add_argument(
        '--num-planning-improvements-per-direct-improvement',
        type=int,
        help='Number of planning improvements to make for each direct improvement.'
    )

    train_function_arg_parser.add_argument(
        '--new-checkpoint-path',
        type=str,
        help='New checkpoint path.'
    )

    parsed_train_function_args, unparsed_args = train_function_arg_parser.parse_known_args(unparsed_args)

    # convert boolean strings to bools
    if parsed_train_function_args.update_upon_every_visit is not None:
        parsed_train_function_args.update_upon_every_visit = parsed_train_function_args.update_upon_every_visit == 'True'

    if parsed_train_function_args.make_final_policy_greedy is not None:
        parsed_train_function_args.make_final_policy_greedy = parsed_train_function_args.make_final_policy_greedy == 'True'

    train_function = import_function(parsed_args.train_function)

    # filter parsed arguments to those accepted by the training function that will be called
    # noinspection PyUnresolvedReferences
    train_function_arg_names = train_function.__code__.co_varnames
    train_function_args = {
        arg: v
        for arg, v in vars(parsed_train_function_args).items()
        if arg in train_function_arg_names
    }

    random_state = RandomState(12345)

    agent = None

    if parsed_args.agent is not None:
        agent_class = load_class(parsed_args.agent)
        agents, unparsed_args = agent_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

        if len(agents) != 1:
            raise ValueError('Training is only supported for single agents. Please specify one agent.')

        agent = agents[0]
        train_function_args['agent'] = agent

    if parsed_args.environment is not None:
        environment_class = load_class(parsed_args.environment)
        train_function_args['environment'], unparsed_args = environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

    if len(unparsed_args) > 0:
        raise ValueError(f'Unparsed arguments remain:  {unparsed_args}')

    if parsed_args.train:
        train_function(
            **train_function_args
        )
    elif parsed_args.resume_train:
        agent = resume_from_checkpoint(
            resume_function=train_function,
            **train_function_args
        )
    else:
        raise ValueError('Unknown trainer action.')

    print('Training complete.')

    if agent is None:
        warnings.warn('No agent resulting at end of training. Nothing to save.')
    else:
        with open(os.path.expanduser(parsed_args.save_agent_path), 'wb') as f:
            pickle.dump(agent, f)

        print(f'Saved agent to {parsed_args.save_agent_path}')

    return (
        train_function_args['new_checkpoint_path'] if 'new_checkpoint_path' in train_function_args else train_function_args['checkpoint_path'],
        parsed_args.save_agent_path
    )


if __name__ == '__main__':
    run(sys.argv[1:])
