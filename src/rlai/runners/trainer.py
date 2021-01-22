import os
import pickle
import sys
import warnings
from typing import List, Tuple, Optional

from numpy.random import RandomState

from rlai.gpi.utils import resume_from_checkpoint
from rlai.meta import rl_text
from rlai.utils import import_function, load_class, get_base_argument_parser, parse_arguments


@rl_text(chapter='Training and Running Agents', page=1)
def run(
        args: List[str] = None
) -> Tuple[Optional[str], str]:
    """
    Train an agent in an environment.

    :param args: Arguments.
    :returns: 2-tuple of the checkpoint path (if any) and the saved agent path.
    """

    random_state = RandomState(12345)

    parser = get_base_argument_parser(
        prog='rlai train',
        description='Train an agent in an environment.'
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
        '--planning-environment',
        type=str,
        help='Fully-qualified type name of planning environment to train agent in.'
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

    parsed_args, unparsed_args = parse_arguments(parser, args)

    train_function_args = {}

    # load environment
    if parsed_args.environment is not None:
        environment_class = load_class(parsed_args.environment)
        train_function_args['environment'], unparsed_args = environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )

    # load planning environment
    if parsed_args.planning_environment is not None:
        planning_environment_class = load_class(parsed_args.planning_environment)
        train_function_args['planning_environment'], unparsed_args = planning_environment_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state
        )
    else:
        train_function_args['planning_environment'] = None

    # get training function and parse its arguments
    train_function = None
    if parsed_args.train_function is not None:

        train_function = import_function(parsed_args.train_function)

        train_function_arg_parser = get_base_argument_parser(prog=parsed_args.train_function)

        # get argument names expected by training function
        # noinspection PyUnresolvedReferences
        train_function_arg_names = train_function.__code__.co_varnames[:train_function.__code__.co_argcount]

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
            if var_name in train_function_arg_names:
                train_function_arg_parser.add_argument(
                    name,
                    **kwargs
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
            '--epsilon',
            type=float,
            help='Total probability mass to allocate across all policy actions.'
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
            help='Fully-qualified type name of state-action value estimator to use.'
        )

        parsed_train_function_args, unparsed_args = parse_arguments(train_function_arg_parser, unparsed_args)

        # convert boolean strings to bools
        if hasattr(parsed_train_function_args, 'update_upon_every_visit') and parsed_train_function_args.update_upon_every_visit is not None:
            parsed_train_function_args.update_upon_every_visit = parsed_train_function_args.update_upon_every_visit == 'True'

        if hasattr(parsed_train_function_args, 'make_final_policy_greedy') and parsed_train_function_args.make_final_policy_greedy is not None:
            parsed_train_function_args.make_final_policy_greedy = parsed_train_function_args.make_final_policy_greedy == 'True'

        train_function_args.update(vars(parsed_train_function_args))

        # initialize state-action value estimator if one is given
        if hasattr(parsed_train_function_args, 'q_S_A') and parsed_train_function_args.q_S_A is not None:
            estimator_class = load_class(parsed_train_function_args.q_S_A)
            train_function_args['q_S_A'], unparsed_args = estimator_class.init_from_arguments(
                unparsed_args,
                random_state=random_state,
                environment=train_function_args['environment'],
                epsilon=train_function_args['epsilon']
            )

    agent = None

    if parsed_args.agent is not None:

        agent_class = load_class(parsed_args.agent)
        agents, unparsed_args = agent_class.init_from_arguments(
            args=unparsed_args,
            random_state=random_state,
            pi=None if train_function_args.get('q_S_A') is None else train_function_args['q_S_A'].get_initial_policy()
        )

        if len(agents) != 1:
            raise ValueError('Training is only supported for single agents. Please specify one agent.')

        agent = agents[0]
        train_function_args['agent'] = agent

    if '--help' in unparsed_args:
        unparsed_args.remove('--help')

    if len(unparsed_args) > 0:
        raise ValueError(f'Unparsed arguments remain:  {unparsed_args}')

    if train_function is None:
        warnings.warn('No training function specified. Cannot train.')
    else:

        # warn user now, as training could take a long time and it'll be wasted effort if the agent is not saved.
        if parsed_args.save_agent_path is None:
            warnings.warn('No --save-agent-path has been specified, so no agent will be saved after training.')

        # resumption will return agent
        if parsed_args.resume_train:
            agent = resume_from_checkpoint(
                resume_function=train_function,
                **train_function_args
            )

        # fresh training initializes agent above
        else:
            train_function(
                **train_function_args
            )

            train_function_args['environment'].close()

        print('Training complete.')

        # try to save agent
        if agent is None:
            warnings.warn('No agent resulting at end of training. Nothing to save.')
        elif parsed_args.save_agent_path is None:
            warnings.warn('No --save-agent-path specified. Not saving agent.')
        else:
            with open(os.path.expanduser(parsed_args.save_agent_path), 'wb') as f:
                pickle.dump(agent, f)

            print(f'Saved agent to {parsed_args.save_agent_path}')

    return train_function_args.get('checkpoint_path'), parsed_args.save_agent_path


if __name__ == '__main__':
    run(sys.argv[1:])
