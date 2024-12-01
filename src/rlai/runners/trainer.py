import logging
import os
import pickle
import sys
import warnings
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Callable, Dict

from numpy.random import RandomState

from rlai.docs import rl_text
from rlai.gpi.utils import resume_from_checkpoint
from rlai.policy_gradient.policies import ParameterizedPolicy
from rlai.utils import (
    import_function,
    load_class,
    get_base_argument_parser,
    parse_arguments,
    RunThreadManager
)


@rl_text(chapter='Training and Running Agents', page=1)
def run(
        args: List[str],
        thread_manager: Optional[RunThreadManager] = None,
        train_function_args_callback: Optional[Callable[[Dict], None]] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    Train an agent in an environment.

    :param args: Arguments.
    :param thread_manager: Thread manager for the thread that is executing the current function. If None, then training
    will continue until termination criteria (e.g., number of iterations) are met. If not None, then the passed
    manager will be waited upon before starting each iteration. If the manager blocks, then another thread will need to
    clear the manager before the iteration continues. If the manager aborts, then this function will return as soon as
    possible.
    :param train_function_args_callback: A callback function to be called with the arguments that will be passed to the
    training function. This gives the caller an opportunity to grab references to the internal arguments that will be
    used in training. For example, plotting from the Jupyter Lab interface grabs the state-action value estimator
    (q_S_A) from the passed dictionary to use in updating its plots. This callback is only called for fresh training. It
    is not called when resuming from a checkpoint.
    :returns: 2-tuple of the checkpoint path (if any) and the saved agent path (if any).
    """

    # initialize with flag set if not passed, so that execution will not block. since the caller will not hold a
    # reference to the manager, it cannot be cleared and execution will never block.
    if thread_manager is None:
        thread_manager = RunThreadManager(True)

    parser = get_argument_parser_for_run()
    parsed_args, unparsed_args = parse_arguments(parser, args)

    if parsed_args.train_function is None:
        raise ValueError('No training function specified. Cannot train.')

    if parsed_args.random_seed is None:
        if parsed_args.resume:
            random_state = None
        else:
            raise ValueError(
                'It is an error to start fresh training without a random seed, since results will not be replicable. '
                'Pass the seed with the --random-seed argument.'
            )
    else:
        random_state = RandomState(parsed_args.random_seed)

    # warn user, as training could take a long time, and it'll be wasted effort if the agent is not saved.
    if parsed_args.save_agent_path is None:
        warnings.warn('No --save-agent-path has been specified, so no agent will be saved after training.')

    # load training function and parse any arguments that it requires
    train_function = import_function(parsed_args.train_function)
    train_function_arg_parser = get_argument_parser_for_train_function(parsed_args.train_function)
    parsed_train_function_args, unparsed_args = parse_arguments(train_function_arg_parser, unparsed_args)

    train_function_args = {
        'thread_manager': thread_manager,
        **vars(parsed_train_function_args)
    }

    # convert boolean strings to booleans
    if train_function_args.get('update_upon_every_visit', None) is not None:
        train_function_args['update_upon_every_visit'] = train_function_args['update_upon_every_visit'] == 'True'

    if train_function_args.get('make_final_policy_greedy', None) is not None:
        train_function_args['make_final_policy_greedy'] = train_function_args['make_final_policy_greedy'] == 'True'

    if train_function_args.get('plot_state_value', None) is not None:
        train_function_args['plot_state_value'] = train_function_args['plot_state_value'] == 'True'

    # load environment
    if train_function_args.get('environment', None) is not None:
        assert random_state is not None
        environment_class = load_class(train_function_args['environment'])
        train_function_args['environment'], unparsed_args = environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

    # load planning environment
    if train_function_args.get('planning_environment', None) is not None:
        assert random_state is not None
        planning_environment_class = load_class(train_function_args['planning_environment'])
        train_function_args['planning_environment'], unparsed_args = planning_environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

    # load agent
    if train_function_args.get('agent', None) is not None:
        assert random_state is not None
        agent_class = load_class(train_function_args['agent'])
        agents, unparsed_args = agent_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            environment=train_function_args['environment']
        )

        if len(agents) != 1:
            raise Exception(f'Expected exactly 1 agent, but got {len(agents)}.')

        agent = agents[0]
        train_function_args['agent'] = agent
    else:
        agent = None

    if '--help' in unparsed_args:
        unparsed_args.remove('--help')

    if len(unparsed_args) > 0:
        raise ValueError(f'Unparsed arguments remain:  {unparsed_args}')

    # resumption will return a trained version of the agent contained in the checkpoint file
    if parsed_args.resume:
        new_checkpoint_path, agent = resume_from_checkpoint(
            resume_function=train_function,
            **train_function_args
        )

    # fresh training will train the agent that was initialized above and passed in
    else:

        if train_function_args_callback is not None:
            train_function_args_callback(train_function_args)

        new_checkpoint_path = train_function(
            **train_function_args
        )

        train_function_args['environment'].close()

        if isinstance(agent.pi, ParameterizedPolicy):
            agent.pi.close()

    logging.info('Training complete.')

    # try to save agent
    if agent is None:  # pragma no cover
        warnings.warn('No agent resulting at end of training. Nothing to save.')
    elif parsed_args.save_agent_path is None:
        warnings.warn('No --save-agent-path specified. Not saving agent.')
    else:
        with open(os.path.expanduser(parsed_args.save_agent_path), 'wb') as f:
            pickle.dump(agent, f)

        logging.info(f'Saved agent to {parsed_args.save_agent_path}')

    return new_checkpoint_path, parsed_args.save_agent_path


def get_argument_parser_for_run() -> ArgumentParser:
    """
    Get argument parser for values used in the run function.

    :return: Argument parser.
    """

    parser = get_base_argument_parser(
        prog='rlai train',
        description='Train an agent in an environment.'
    )

    parser.add_argument(
        '--train-function',
        type=str,
        help='Fully-qualified type name of function to use for training the agent.'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Pass this flag to resume training an agent from a previously saved checkpoint path.'
    )

    parser.add_argument(
        '--save-agent-path',
        type=str,
        help='Path to store resulting agent to.'
    )

    parser.add_argument(
        '--random-seed',
        type=int,
        help='Random seed. Omit to generate an arbitrary random seed.'
    )

    return parser


def get_argument_parser_for_train_function(
        function_name: str
) -> ArgumentParser:
    """
    Get argument parser for a train function.

    :param function_name: Function name.
    :return: Argument parser.
    """

    argument_parser = get_base_argument_parser(prog=function_name)

    function = import_function(function_name)

    # get argument names defined by the specified training function
    # noinspection PyUnresolvedReferences
    function_arg_names = function.__code__.co_varnames[:function.__code__.co_argcount]

    def add_argument(
            name: str,
            **kwargs
    ):
        """
        Filter arguments to those defined by the function before adding them to the argument parser.

        :param name: Argument name.
        :param kwargs: Other arguments.
        """

        var_name = name.lstrip('-').replace('-', '_')
        if var_name in function_arg_names:
            argument_parser.add_argument(
                name,
                **kwargs
            )

    # add the superset of all arguments used across all training function. the filter will only retain those allowed.

    add_argument(
        '--agent',
        type=str,
        help='Fully-qualified type name of agent to train.'
    )

    add_argument(
        '--environment',
        type=str,
        help='Fully-qualified type name of environment to train agent in.'
    )

    add_argument(
        '--planning-environment',
        type=str,
        help='Fully-qualified type name of planning environment to train agent in.'
    )

    add_argument(
        '--num-improvements',
        type=int,
        help='Number of improvements.'
    )

    add_argument(
        '--num-episodes-per-improvement',
        type=int,
        help='Number of episodes per improvement.'
    )

    add_argument(
        '--num-episodes',
        type=int,
        help='Number of episodes.'
    )

    add_argument(
        '--num-updates-per-improvement',
        type=int,
        help='Number of state-action value updates per policy improvement.'
    )

    add_argument(
        '--update-upon-every-visit',
        type=str,
        choices=['True', 'False'],
        help='Whether to update values upon each visit to a state or state-action pair.'
    )

    add_argument(
        '--alpha',
        type=float,
        help='Step size.'
    )

    add_argument(
        '--make-final-policy-greedy',
        type=str,
        choices=['True', 'False'],
        help='Whether to make the final policy greedy after training is complete.'
    )

    add_argument(
        '--num-improvements-per-plot',
        type=int,
        help='Number of improvements per plot.'
    )

    add_argument(
        '--num-episodes-per-policy-update-plot',
        type=int,
        help='Number of episodes per policy-update plot.'
    )

    add_argument(
        '--policy-update-plot-pdf-directory',
        type=str,
        help='Directory in which to store plot PDFs, or None to display them directly.'
    )

    add_argument(
        '--num-warmup-episodes',
        type=int,
        help='Number of warmup episodes.'
    )

    add_argument(
        '--num-improvements-per-checkpoint',
        type=int,
        help='Number of improvements per checkpoint.'
    )

    add_argument(
        '--num-episodes-per-checkpoint',
        type=int,
        help='Number of episodes per checkpoint.'
    )

    add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to checkpoint file.'
    )

    add_argument(
        '--mode',
        type=str,
        help='Temporal difference evaluation mode (SARSA, Q_LEARNING, EXPECTED_SARSA).'
    )

    add_argument(
        '--n-steps',
        type=int,
        help='N-step update value.'
    )

    add_argument(
        '--pdf-save-path',
        type=str,
        help='Path where a PDF of all plots is to be saved.'
    )

    add_argument(
        '--plot-state-value',
        type=str,
        choices=['True', 'False'],
        help='Whether to plot the state value.'
    )

    add_argument(
        '--training-pool-directory',
        type=str,
        help='Path to directory in which to store pooled training runs.'
    )

    add_argument(
        '--training-pool-count',
        type=int,
        help='Number of runners in the training pool.'
    )

    add_argument(
        '--training-pool-iterate-episodes',
        type=int,
        help='Number of episodes per training pool iteration.'
    )

    add_argument(
        '--training-pool-evaluate-episodes',
        type=int,
        help='Number of episodes to evaluate the agent when iterating the training pool.'
    )

    add_argument(
        '--training-pool-max-iterations-without-improvement',
        type=int,
        help=(
            'Maximum number of training pool iterations to allow before reverting to the best prior agent, or None to '
            'never revert.'
        )
    )

    add_argument(
        '--start-episode',
        type=int,
        help=(
            '1-based episode to start at.'
        )
    )

    add_argument(
        '--start-improvement',
        type=int,
        help=(
            '1-based improvement to start at.'
        )
    )

    return argument_parser


if __name__ == '__main__':  # pragma no cover
    run(sys.argv[1:])
