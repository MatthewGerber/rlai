import logging
import os
import pickle
import sys
import warnings
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Callable, Dict

from numpy.random import RandomState

from rlai.gpi.utils import resume_from_checkpoint
from rlai.meta import rl_text
from rlai.policies.parameterized import ParameterizedPolicy
from rlai.utils import (
    import_function,
    load_class,
    get_base_argument_parser,
    parse_arguments,
    RunThreadManager
)


@rl_text(chapter='Training and Running Agents', page=1)
def run(
        args: List[str] = None,
        thread_manager: RunThreadManager = None,
        train_function_args_callback: Callable[[Dict], None] = None
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
        warnings.warn('No random seed provided to the trainer. Results will not be replicable. Consider passing --random-seed argument.')
        random_state = RandomState()
    else:
        random_state = RandomState(parsed_args.random_seed)

    # warn user, as training could take a long time and it'll be wasted effort if the agent is not saved.
    if parsed_args.save_agent_path is None:
        warnings.warn('No --save-agent-path has been specified, so no agent will be saved after training.')

    initial_policy = None

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

    # load environment
    if train_function_args.get('environment', None) is not None:
        environment_class = load_class(train_function_args['environment'])
        train_function_args['environment'], unparsed_args = environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

    # load planning environment
    if train_function_args.get('planning_environment', None) is not None:
        planning_environment_class = load_class(train_function_args['planning_environment'])
        train_function_args['planning_environment'], unparsed_args = planning_environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

    # load state-action value estimator
    if train_function_args.get('q_S_A', None) is not None:
        estimator_class = load_class(train_function_args['q_S_A'])
        state_action_value_estimator, unparsed_args = estimator_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            environment=train_function_args['environment']
        )
        train_function_args['q_S_A'] = state_action_value_estimator
        initial_policy = state_action_value_estimator.get_initial_policy()

    # load state-value estimator
    if train_function_args.get('v_S', None) is not None:
        estimator_class = load_class(train_function_args['v_S'])
        train_function_args['v_S'], unparsed_args = estimator_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            environment=train_function_args['environment']
        )

    # load parameterized policy
    if train_function_args.get('policy', None) is not None:
        policy_class = load_class(train_function_args['policy'])
        initial_policy, unparsed_args = policy_class.init_from_arguments(
            args=unparsed_args,
            environment=train_function_args['environment']
        )
        train_function_args['policy'] = initial_policy

    # load agent
    if train_function_args.get('agent', None) is not None:
        agent_class = load_class(train_function_args['agent'])
        agent, unparsed_args = agent_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            pi=initial_policy
        )
        train_function_args['agent'] = agent
    else:
        agent = None

    if '--help' in unparsed_args:
        unparsed_args.remove('--help')

    if len(unparsed_args) > 0:
        raise ValueError(f'Unparsed arguments remain:  {unparsed_args}')

    # resumption will return a trained version of the agent contained in the checkpoint file
    if parsed_args.resume:
        agent = resume_from_checkpoint(
            resume_function=train_function,
            **train_function_args
        )

    # fresh training will train the agent that was initialized above and passed in
    else:

        if train_function_args_callback is not None:
            train_function_args_callback(train_function_args)

        train_function(
            **train_function_args
        )

        train_function_args['environment'].close()

        if isinstance(initial_policy, ParameterizedPolicy):
            initial_policy.close()

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

    return train_function_args.get('checkpoint_path'), parsed_args.save_agent_path


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

    # get argument names actuall expected by the specified training function
    # noinspection PyUnresolvedReferences
    actual_arg_names = function.__code__.co_varnames[:function.__code__.co_argcount]

    def filter_add_argument(
            name: str,
            **kwargs
    ):
        """
        Filter arguments to those defined by the function before adding them to the argument parser.

        :param name: Argument name.
        :param kwargs: Other arguments.
        """

        var_name = name.lstrip('-').replace('-', '_')
        if var_name in actual_arg_names:
            argument_parser.add_argument(
                name,
                **kwargs
            )

    # attempt to add the superset of all arguments used across all training function. the filter will only retain those
    # that are actually allowed.

    filter_add_argument(
        '--agent',
        type=str,
        help='Fully-qualified type name of agent to train.'
    )

    filter_add_argument(
        '--environment',
        type=str,
        help='Fully-qualified type name of environment to train agent in.'
    )

    filter_add_argument(
        '--planning-environment',
        type=str,
        help='Fully-qualified type name of planning environment to train agent in.'
    )

    filter_add_argument(
        '--policy',
        type=str,
        help='Fully-qualified type name of policy to use (for policy gradient methods).'
    )

    filter_add_argument(
        '--num-improvements',
        type=int,
        help='Number of improvements.'
    )

    filter_add_argument(
        '--num-episodes-per-improvement',
        type=int,
        help='Number of episodes per improvement.'
    )

    filter_add_argument(
        '--num-episodes',
        type=int,
        help='Number of episodes.'
    )

    filter_add_argument(
        '--num-updates-per-improvement',
        type=int,
        help='Number of state-action value updates per policy improvement.'
    )

    filter_add_argument(
        '--update-upon-every-visit',
        type=str,
        choices=['True', 'False'],
        help='Whether or not to update values upon each visit to a state or state-action pair.'
    )

    filter_add_argument(
        '--alpha',
        type=float,
        help='Step size.'
    )

    filter_add_argument(
        '--make-final-policy-greedy',
        type=str,
        choices=['True', 'False'],
        help='Whether or not to make the final policy greedy after training is complete.'
    )

    filter_add_argument(
        '--num-improvements-per-plot',
        type=int,
        help='Number of improvements per plot.'
    )

    filter_add_argument(
        '--num-improvements-per-checkpoint',
        type=int,
        help='Number of improvements per checkpoint.'
    )

    filter_add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to checkpoint file.'
    )

    filter_add_argument(
        '--mode',
        type=str,
        help='Temporal difference evaluation mode (SARSA, Q_LEARNING, EXPECTED_SARSA).'
    )

    filter_add_argument(
        '--n-steps',
        type=int,
        help='N-step update value.'
    )

    filter_add_argument(
        '--q-S-A',
        type=str,
        help='Fully-qualified type name of state-action value estimator to use (for action-value methods).'
    )

    filter_add_argument(
        '--v-S',
        type=str,
        help='Fully-qualified type name of state-value estimator to use (for policy gradient methods).'
    )

    filter_add_argument(
        '--pdf-save-path',
        type=str,
        help='Path where a PDF of all plots is to be saved.'
    )

    filter_add_argument(
        '--plot-state-value',
        action='store_true',
        help='Pass this flag to plot the state value.'
    )

    return argument_parser


if __name__ == '__main__':  # pragma no cover
    run(sys.argv[1:])
