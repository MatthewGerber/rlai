# MuJoCo Swimming Worm
* Content
{:toc}

# Introduction

You can read more about this environment [here](https://gym.openai.com/envs/Swimmer-v2/). Many of the 
issues involved in solving this environment are addressed in the 
[continuous mountain car](mountain_car_continuous.md) and 
[continuous lunar lander](lunar_lander_continuous.md) case studies, so we will focus here on details specific to the
MuJoCo swimming worm environment. There are not many. Except for a minor change of learning rates, this problem was 
solved with a direct application of the same policy gradient method explored in the prior case studies. The training 
time was significantly greater, but the environment itself is simpler:  no fuel levels and a simpler reward function.

# Development
A few key points of development are worth mentioning.

### Feature Space

### Baseline and Policy Gradient Learning Rates

# Training
The following command trains an agent for the MuJoCo swimming worm environment using policy gradient optimization 
with a baseline state-value estimator:

```
rlai train 
```

The argument are explained below.

### RLAI
* `train`:  Train the agent. 
* `--random-seed 12345`:  For reproducibility.

### Agent
* `--agent rlai.agents.mdp.StochasticMdpAgent`:  Standard stochastic MDP agent. 
* `--gamma 1.0`:  Do not discount.

### Environment
* `--environment rlai.environments.openai_gym.Gym`:  Environment class.
* `--gym-id Swimmer-v2`:  OpenAI Gym environment identifier.
* `--render-every-nth-episode 100`:  Render a video every 100 episodes.
* `--video-directory ~/Desktop/swimmer_videos`:  Where to store rendered videos.
* `--force`:  Overwrite videos in the video directory.
* `--plot-environment`:  Show a real-time plot of state and reward values.
* `--T 500`:  Limit episodes to 500 steps.

### Training Function and Episodes
* `--train-function rlai.policy_gradient.monte_carlo.reinforce.improve`:  Run the REINFORCE policy gradient optimization
algorithm.
* `--num-episodes 50000`:  Run 50000 episodes.

### Baseline State-Value Estimator
* `--plot-state-value True`:  Show a real-time plot of the estimated baseline state value.
* `--v-S rlai.v_S.function_approximation.estimators.ApproximateStateValueEstimator`:  Baseline state-value estimator.  
* `--feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor`:  Feature extractor for the
baseline state-value estimator.
* `--function-approximation-model rlai.models.sklearn.SKLearnSGD`:  Use SKLearn's SGD for the baseline state-value 
estimator.
* `--loss squared_loss`:  Use a squared loss within the baseline state-value estimator.
* `--sgd-alpha 0.0`:  Do not use regularization.
* `--learning-rate constant`:  Use a constant learning rate schedule.
* `--eta0 0.0001`:  Learning rate.

### Policy
* `--policy rlai.policies.parameterized.continuous_action.ContinuousActionBetaDistributionPolicy`:  Use the beta
distribution to model the action-density distribution within the policy.
* `--policy-feature-extractor rlai.environments.openai_gym.SignedCodingFeatureExtractor`:  Feature extractor
for the policy gradient optimizer.
* `--plot-policy`:  Show a real-time display of the action that is selected at each step.
* `--alpha 0.0001`:  Learning rate for policy gradient updates.
* `--update-upon-every-visit True`:  Update the policy's action-density distribution every time a state is encountered
  (as opposed to the first visit only).

### Output
* `--save-agent-path ~/Desktop/swimmer_agent.pickle`:  Where to save the final agent.

# Results

The following sequence of videos shows the progression of policies:

{% include youtubePlayer.html id="" %}
