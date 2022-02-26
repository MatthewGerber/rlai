#!/bin/bash

rm -f kill_swimmer_pool.sh

eta_values=(0.00001 0.00001 0.0001 0.0001 0.001 0.001 0.01 0.01 0.1 0.1)

alpha_values=(0.000001 0.000001 0.00001 0.00001 0.0001 0.0001 0.001 0.001 0.01 0.01)

for n in $(seq 0 9); do
    rlai train --random-seed "1234${n}" --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1.0 --environment rlai.environments.openai_gym.Gym --gym-id Swimmer-v2 --render-every-nth-episode 200 --video-directory "$HOME/Desktop/swimmer_videos_${n}" --force --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 20000 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 "${eta_values[$n]}" --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor --alpha "${alpha_values[$n]}" --update-upon-every-visit True --num-episodes-per-checkpoint 1000 --checkpoint-path $"$HOME/Desktop/swimmer_${n}_checkpoint.pickle" --save-agent-path "$HOME/Desktop/swimmer_agent_${n}.pickle" --log INFO --training-pool-directory ~/Desktop/swimmer_pool --training-pool-batch-size 20 &
    echo "sudo kill -9 $!" >> kill_swimmer_pool.sh
done