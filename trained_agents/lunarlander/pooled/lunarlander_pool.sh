#!/bin/bash

rm -f kill_lunarlander_pool.sh
rm -f lunarlander_pool.log

num_runners=10
eta_values=(0.0001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)
alpha_values=(0.0001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001 0.001)

for n in $(seq 0 $((num_runners - 1))); do
    rlai train --random-seed "1234${n}" --agent rlai.agents.mdp.StochasticMdpAgent --gamma 1.0 --environment rlai.environments.openai_gym.Gym --gym-id LunarLanderContinuous-v2 --render-every-nth-episode 500 --video-directory "$HOME/Desktop/lunarlander_videos_${n}" --force --T 500 --train-function rlai.policy_gradient.monte_carlo.reinforce.improve --num-episodes 10000 --v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator --feature-extractor rlai.environments.openai_gym.ContinuousLunarLanderFeatureExtractor --function-approximation-model rlai.models.sklearn.SKLearnSGD --loss squared_loss --sgd-alpha 0.0 --learning-rate constant --eta0 "${eta_values[$n]}" --policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy --policy-feature-extractor rlai.environments.openai_gym.ContinuousLunarLanderFeatureExtractor --alpha "${alpha_values[$n]}" --update-upon-every-visit True --num-episodes-per-checkpoint 2000 --checkpoint-path $"$HOME/Desktop/lunarlander_${n}_checkpoint.pickle" --save-agent-path "$HOME/Desktop/lunarlander_agent_${n}.pickle" --log INFO --training-pool-directory ~/Desktop/lunarlander_pool --training-pool-count ${num_runners} --training-pool-iterate-episodes 10 --training-pool-evaluate-episodes 10 >>lunarlander_pool.log 2>&1 &
    echo "sudo kill -9 $!" >> kill_lunarlander_pool.sh
done