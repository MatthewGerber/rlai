#!/bin/bash

rm -f kill_swimmer_pool.sh
rm -f swimmer_pool.log

num_runners=10
eta_values=(0.0001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
alpha_values=(0.00001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001)

for n in $(seq 0 $((num_runners - 1))); do
    rlai train --random-seed "1234${n}" --agent rlai.policy_gradient.ParameterizedMdpAgent --gamma 1.0 --environment rlai.core.environments.gymnasium.Gym --gym-id Swimmer-v5 --render-every-nth-episode 100 --video-directory "$HOME/Desktop/swimmer_videos_${n}" --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 4000 --v-S rlai.state_value.function_approximation.ApproximateStateValueEstimator --feature-extractor rlai.core.environments.gymnasium.SignedCodingFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 "${eta_values[$n]}" --policy rlai.policy_gradient.policies.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.core.environments.gymnasium.SignedCodingFeatureExtractor --alpha "${alpha_values[$n]}" --update-upon-every-visit True --num-episodes-per-checkpoint 1000 --checkpoint-path $"$HOME/Desktop/swimmer_${n}_checkpoint.pickle" --save-agent-path "$HOME/Desktop/swimmer_agent_${n}.pickle" --log INFO --training-pool-directory ~/Desktop/swimmer_pool --training-pool-count ${num_runners} --training-pool-iterate-episodes 10 --training-pool-evaluate-episodes 10 --num-episodes-per-policy-update-plot 100 --policy-update-plot-pdf-directory $"$HOME/Desktop/swimmer_${n}_plots" >>swimmer_pool.log 2>&1 &
    echo "sudo kill -9 $!" >> kill_swimmer_pool.sh
done