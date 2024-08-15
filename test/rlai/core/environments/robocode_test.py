import json
import os
import pickle
import shlex
import socket
import tempfile
import time
from threading import Thread

import numpy as np
import pytest

from rlai.core.environments.robocode import RobocodeFeatureExtractor
from rlai.runners.trainer import run


@pytest.mark.will_fail_gh
def test_learn():
    """
    Test.
    """

    # set the following to True to update the fixture. then uncomment some stuff in
    # rlai.core.environments.network.TcpMdpEnvironment.read_from_client in order to update the test fixture. finally,
    # start this test and then start the robocode game. run a battle for 10 rounds to complete the fixture update.
    update_fixture = False

    robocode_port = 54321

    robocode_mock_thread = None

    if not update_fixture:

        with open(f'{os.path.dirname(__file__)}/fixtures/test_robocode.pickle', 'rb') as load_f:
            state_sequence, fixture_pi, fixture_q_S_A = pickle.load(load_f)

        # set up a mock robocode game that sends state sequence
        def robocode_mock():

            # wait for environment to start up and listen for connections
            time.sleep(5)

            t = 0
            while t < len(state_sequence):

                # start episode by connecting
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('127.0.0.1', robocode_port))

                    # noinspection PyBroadException
                    try:

                        while t < len(state_sequence):

                            # send the current game state in the sequence
                            state_dict_json = state_sequence[t]
                            s.sendall(state_dict_json.encode('utf-8'))
                            t += 1

                            # receive next action
                            s.recv(99999999)

                            # if the next state starts a new episode, then break.
                            if t < len(state_sequence):
                                next_state_dict = json.loads(state_sequence[t])
                                if next_state_dict['state']['time'] == 0:
                                    break

                    # if environment closes connection during receive, it ends the episode.
                    except Exception:  # pragma no cover
                        pass

        robocode_mock_thread = Thread(target=robocode_mock)
        robocode_mock_thread.start()

    # run training and load resulting agent
    agent_path = tempfile.NamedTemporaryFile(delete=False).name
    cmd = f'--random-seed 12345 --agent rlai.core.environments.robocode.RobocodeAgent --gamma 0.95 --environment rlai.core.environments.robocode.RobocodeEnvironment --port {robocode_port} --bullet-power-decay 0.75 --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --n-steps 50 --num-improvements 10 --num-episodes-per-improvement 1 --num-updates-per-improvement 1 --epsilon 0.25 --q-S-A rlai.gpi.state_action_value.function_approximation.ApproximateStateActionValueEstimator --function-approximation-model rlai.gpi.state_action_value.function_approximation.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.0001 --feature-extractor rlai.core.environments.robocode.RobocodeFeatureExtractor --scanned-robot-decay 0.75 --make-final-policy-greedy True --num-improvements-per-plot 100 --save-agent-path {agent_path} --log DEBUG'
    run(shlex.split(cmd))

    if not update_fixture:
        assert robocode_mock_thread is not None
        robocode_mock_thread.join()

    with open(agent_path, 'rb') as load_f:
        agent = pickle.load(load_f)

    # if we're updating the test fixture, then save the state sequence and resulting policy to disk.
    if update_fixture:  # pragma no cover
        with open(os.path.expanduser('~/Desktop/state_sequence.txt'), 'r') as txt_f:
            state_sequence = txt_f.readlines()
        with open(f'{os.path.dirname(__file__)}/fixtures/test_robocode.pickle', 'wb') as dump_f:
            pickle.dump((state_sequence, agent.pi, agent.pi.estimator), dump_f)
    else:
        assert np.allclose(agent.pi.estimator.model.sklearn_sgd.model.coef_, fixture_q_S_A.model.sklearn_sgd.model.coef_)
        assert np.allclose(agent.pi.estimator.model.sklearn_sgd.model.intercept_, fixture_q_S_A.model.sklearn_sgd.model.intercept_)


def test_feature_extractor():
    """
    Test.
    """

    assert RobocodeFeatureExtractor.is_clockwise_move(0, 1)
    assert RobocodeFeatureExtractor.is_clockwise_move(-1, 1)
    assert not RobocodeFeatureExtractor.is_clockwise_move(1, 1)
    assert RobocodeFeatureExtractor.get_shortest_degree_change(1, -1) == -2
    assert RobocodeFeatureExtractor.get_shortest_degree_change(-1, 1) == 2
    assert RobocodeFeatureExtractor.get_shortest_degree_change(1, 1) == 0
    assert RobocodeFeatureExtractor.normalize(366) == 6
    assert RobocodeFeatureExtractor.normalize(-5) == 355
