import os
import pickle
import tempfile

from rlai.runners.trainer import run


def test_run():

    run_args_list = [
        f'--train --agent rlai.agents.mdp.StochasticMdpAgent --continuous-state-discretization-resolution 0.1 --gamma 1 --environment rlai.environments.openai_gym.Gym --gym-id CartPole-v1 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --n-steps 10 --num-improvements 3 --num-episodes-per-improvement 5 --alpha 0.1 --epsilon 0.01 --state-action-value-estimator rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 3 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--train --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.mdp.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.TrajectorySamplingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --state-action-value-estimator rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}',
        f'--train --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1 --environment rlai.environments.mdp.Gridworld --id example_4_1 --planning-environment rlai.environments.mdp.PrioritizedSweepingMdpPlanningEnvironment --num-planning-improvements-per-direct-improvement 10 --priority-theta 0.1 --T-planning 50 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 10 --num-episodes-per-improvement 5 --epsilon 0.01 --state-action-value-estimator rlai.value_estimation.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-checkpoint 10 --checkpoint-path {tempfile.NamedTemporaryFile(delete=False).name} --save-agent-path {tempfile.NamedTemporaryFile(delete=False).name}'
    ]

    run_checkpoint_agent = {}

    for run_args in run_args_list:

        checkpoint_path, agent_path = run(run_args.split())

        if checkpoint_path is None:
            checkpoint = None
        else:
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)

        with open(agent_path, 'rb') as f:
            agent = pickle.load(f)

        run_checkpoint_agent[run_args] = checkpoint, agent

    # uncomment the following line and run test to update fixture
    # with open(f'{os.path.dirname(__file__)}/fixtures/trainer_test.pickle', 'wb') as f:
    #     pickle.dump(run_checkpoint_agent, f)

    with open(f'{os.path.dirname(__file__)}/fixtures/trainer_test.pickle', 'rb') as f:
        run_fixture = pickle.load(f)

    assert len(run_args_list) == len(run_fixture.keys())

    for run_args, run_args_fixture in zip(run_args_list, run_fixture.keys()):

        print(f'Checking test results for run {run_args}...', end='')

        checkpoint, agent = run_checkpoint_agent[run_args]
        checkpoint_fixture, agent_fixture = run_fixture[run_args_fixture]

        assert checkpoint['q_S_A'] == checkpoint_fixture['q_S_A']
        assert agent.pi, agent_fixture.pi

        print('passed.')
