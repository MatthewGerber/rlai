[Home](../index.md) > Inverted Pendulum
* Content
{:toc}
  
# Introduction
The inverted pendulum is also known as cart-pole balancing, where the goal is to keep a bottom-hinged pole balanced for
as long as possible by moving a cart left or right. Imagine balancing a broom with the handle's end in your open palm. 
You can read more about this environment [here](https://gymnasium.farama.org/environments/classic_control/cart_pole/). 
Below is an example of running a random (untrained) agent in this environment. The episode terminates almost immediately
as the agent loses balance control.

{% include youtubePlayer.html id="rGnf9CFwD7M" %}

# Tabular State-Action Value Function
This section describes training and results for an agent using tabular methods for state-action value estimation. The
primary challenge with this approach is to simultaneously (1) discretize the continuous state space to a sufficiently 
fine resolution, thereby resolving state-action pairs sufficient for control, and (2) estimate the state-action value 
function in a practically feasible number of time steps and within memory constraints. These objectives are opposed, 
with increases in resolution making estimation more expensive. The tradeoff is unavoidable with tabular methods. A 
later section of this case study will eliminate discretization of the state space by imposing a parametric form on the 
state-action value function. This change will eliminate memory constraints, but it will introduce several new 
challenges. First, the tabular approach...

## Training
Train a control agent for the inverted pendulum environment with the following command.
```
rlai train --agent rlai.gpi.state_action_value.ActionValueMdpAgent --continuous-state-discretization-resolution 0.1 --gamma 1 --environment rlai.core.environments.gymnasium.Gym --gym-id CartPole-v1 --render-every-nth-episode 5000 --video-directory ~/Desktop/cartpole_videos --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode Q_LEARNING --num-improvements 5000 --num-episodes-per-improvement 50 --T 1000 --epsilon 0.01 --q-S-A rlai.gpi.state_action_value.tabular.TabularStateActionValueEstimator --make-final-policy-greedy True --num-improvements-per-plot 100 --save-agent-path ~/Desktop/cartpole_agent.pickle
```

Arguments are explained below.

### RLAI
* `train`:  Train the agent.

### Agent  
* `--agent rlai.gpi.state_action_value.ActionValueMdpAgent`:  Standard action-value MDP agent. 
* `--continuous-state-discretization-resolution 0.1`:  Discretize the continuous state space into discrete intervals 
  with resolution 0.1. The methods used here are for discrete-state problems, so some type of discretization of the 
  continuous state space is required.
* `--gamma 1`:  No discount. All state-action pairs in the episode will receive equal credit for the total duration of 
  balancing achieved.
  
### Environment
* `--environment rlai.core.environments.gymnasium.Gym`:  Environment class. 
* `--gym-id CartPole-v1`:  Gym environment identifier.
* `--render-every-nth-episode 5000`:  Render a video every 5000 episodes (100 improvements).
* `--video-directory ~/Desktop/cartpole_videos`:  Where to store rendered videos.
  
### State-Action Value Iteration
* `--train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi`:  Run iterative temporal-differencing 
  on the agent's state-action value function. 
* `--mode Q_LEARNING`:  Use q-learning to bootstrap the value of the next state-action pair. 
* `--num-improvements 5000`:  Number of policy improvements (iterations).
* `--num-episodes-per-improvement 50`:  Number of episodes per improvement.
* `--T 1000`:  Maximum number of time steps per episode. Without this, the episode length will be unconstrained. This 
  can slow down learning in later iterations where the agent has developed a reasonable policy.
* `--epsilon 0.01`:  Probability of behaving randomly at each time step.
  
### Other Parameters
* `--make-final-policy-greedy True`:  After all learning iterations, make the final policy greedy (i.e., `epsilon=0.0`).
* `--num-improvements-per-plot 100`:  Plot training performance every 100 iterations.
* `--save-agent-path ~/Desktop/cartpole_agent.pickle`:  Where to save the final agent.

The training progression is shown below.

![inverted-pendulum-tabular](https://github.com/MatthewGerber/rlai/raw/master/trained_agents/cartpole/tabular/cartpole_training.png)

In the left sub-figure above, the left y-axis shows how long the control agent is able to keep the pole balanced, the 
right y-axis shows the size of the state space, and the x-axis shows improvement iterations for the agent's policy. The 
right sub-figure shows the same reward y-axis but along a time x-axis. Based on the learning trajectory, it seems clear
that the agent would have continued to improve its policy given more time; however, as shown below, the results are 
quite satisfactory after 7 hours of wallclock training time. Note, however, that the value function estimator is 
approaching 10^5 (100000) states for this relatively simple environment.

## Results
The video below shows the trained agent controlling the inverted pendulum. Note how the agent actively controls both the 
vertical balance of the pendulum (ideally upright) and the cart's horizontal position along the track (ideally in the 
middle). The episode ends after the maximum number of time steps is reached, rather than due to a control failure.

{% include youtubePlayer.html id="bnQFT31_WfI" %}

# Parametric State-Action Value Function
As shown above, tabular methods present a conceptually straightforward approach to estimating state-action value 
functions in continuous state-space environments. Practically, the estimation problem is complicated by the need for
very large state spaces and accordingly long training times. This is akin to the challenges presented by nonparametric 
function approximation methods such as k-nearest neighbor (KNN). KNN retains all training samples (it is memory-based), 
and the function is approximated at a point by averaging a subsample around the point. The tabular approach described 
above does not retain all training samples, but it does retain an average value in each of the discretized state-space 
intervals. The number of intervals is bounded only by the range of the state dimensions (which may be unbounded). The 
time and memory challenges in KNN and discretized tabular methods are therefore quite similar.

The approach in this section is to approximate the state-action value function by imposing a parametric form upon it.
We will have far fewer parameters (e.g., 10^2) than the number of discretized intervals in our tabular approach (e.g., 
10^5), but we will consequently need to take great care when defining the parametric form and estimating the parameter
values from experience in the environment. There is nothing novel about this tradeoff; it is exactly the distinction 
between parametric and nonparametric statistical learning methods.

## Training
One of the most challenging aspects of parametric RL is selecting training hyperparameters (i.e., parameters of learning
beyond those of the parametric form, such as step sizes). Very little scientific theory exists to guide a-priori 
setting of hyperparameter values in arbitrary tasks. As a result, significant trial-and-error is usually involved, 
either manual or by automated search. Call it an "art" if you wish, but that might be too generous. This section 
presents training with hyperparameters values that generate a high-performance agent, but keep in mind that significant
manual experimentation was required. A later section will show what happens when these values are changed.

```
rlai train --agent rlai.gpi.state_action_value.ActionValueMdpAgent --gamma 0.95 --environment rlai.core.environments.gymnasium.Gym --gym-id CartPole-v1 --render-every-nth-episode 100 --video-directory ~/Desktop/cartpole_videos --train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi --mode SARSA --num-improvements 15000 --num-episodes-per-improvement 1 --num-updates-per-improvement 1 --epsilon 0.2 --q-S-A rlai.gpi.state_action_value.function_approximation.ApproximateStateActionValueEstimator --function-approximation-model rlai.gpi.state_action_value.function_approximation.models.sklearn.SKLearnSGD --loss squared_error --sgd-alpha 0.0 --learning-rate constant --eta0 0.0001 --feature-extractor rlai.core.environments.gymnasium.CartpoleFeatureExtractor --make-final-policy-greedy True --num-improvements-per-plot 100 --num-improvements-per-checkpoint 100 --checkpoint-path ~/Desktop/cartpole_checkpoint.pickle --save-agent-path ~/Desktop/cartpole_agent.pickle
```

Arguments are explained below (many explanations are given above and not duplicated here).

### RLAI 
* `train`:  Train the agent.

### Agent
* `--agent rlai.gpi.state_action_value.ActionValueMdpAgent`
* `--gamma 0.95`

### Environment
* `--environment rlai.core.environments.gymnasium.Gym`
* `--gym-id CartPole-v1`
* `--render-every-nth-episode 100`
* `--video-directory ~/Desktop/cartpole_videos`

### State-Action Value Iteration
* `--train-function rlai.gpi.temporal_difference.iteration.iterate_value_q_pi`
* `--mode SARSA`
* `--num-improvements 15000`
* `--num-episodes-per-improvement 1`
* `--num-updates-per-improvement 1`
* `--epsilon 0.2`

### State-Action Value Model
* `--q-S-A rlai.gpi.state_action_value.function_approximation.ApproximateStateActionValueEstimator`:  Use function 
approximation.
* `--function-approximation-model rlai.gpi.state_action_value.function_approximation.models.sklearn.SKLearnSGD`:  Use 
scikit-learn's stochastic gradient descent. Documentation for the model and its arguments below can be found 
[here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html).
* `--loss squared_error`
* `--sgd-alpha 0.0`
* `--learning-rate constant`
* `--eta0 0.0001`
* `--feature-extractor rlai.core.environments.gymnasium.CartpoleFeatureExtractor`:  Use the feature extractor specified.

### Other Parameters
* `--make-final-policy-greedy True`
* `--num-improvements-per-plot 100`
* `--num-improvements-per-checkpoint 100`
* `--checkpoint-path ~/Desktop/cartpole_checkpoint.pickle`
* `--save-agent-path ~/Desktop/cartpole_agent.pickle`
  
## Results
The video below shows the trained agent controlling the inverted pendulum. Note how the agent actively controls both the 
vertical balance of the pendulum (ideally upright) and the cart's horizontal position along the track (ideally in the 
middle). The episode ends after the maximum number of time steps is reached, rather than due to a control failure.

{% include youtubePlayer.html id="E76S7YTDoek" %}

Also note that the control obtained here is much tighter than achieved with tabular methods above.

## Discussion
As noted above, obtaining an agent that performs well usually involves significant experimentation, and the parameter
selection above is no exception. It is worth mentioning a few points along the way that seemed to be important.

### Nonlinear Feature Space
The cart-pole environment has four continuous state variables:  position, velocity, pole angle, and pole angular 
velocity. The feature extractor for this environment uses both the 
[raw and squared versions](https://github.com/MatthewGerber/rlai/blob/36b755098e75dd1222a802933075db2ab889b29c/src/rlai/environments/openai_gym.py#L438-L441)
of these state variables.

### Feature Contexts
I struggled for a while with the features described above. The agent simply could not learn a useful state-action value 
function. Then it occurred to me that a linear increase in any of those variables (whether raw or squared) could have 
different implications for the value function depending on the context in which they occurred. For example, the value
of moving the cart to the left when the pole is tilted slightly to the left depends on the pole's angular velocity. If
the pole is already swinging back to upright, then moving the cart left might be unnecessary or even harmful. If the 
pole is swinging left, then moving the cart left is probably a good idea. More generally, the value of a particular
action vis-à-vis a state variable depends on the context, that is, on the other state variables. The approach taken 
here is to form statistical interactions of the state variables with a small set of 
[contexts](https://github.com/MatthewGerber/rlai/blob/36b755098e75dd1222a802933075db2ab889b29c/src/rlai/environments/openai_gym.py#L519) 
that differentiate the state-action values. When fitting the model, the state variables are 
[interacted](https://github.com/MatthewGerber/rlai/blob/36b755098e75dd1222a802933075db2ab889b29c/src/rlai/environments/openai_gym.py#L455) 
with these contexts, essentially forming a one-hot-context encoding of the state variables. Interpreted another way, the 
model learns a separate set of parameters for each context. The one-hot-context encoding is further 
[interacted](https://github.com/MatthewGerber/rlai/blob/36b755098e75dd1222a802933075db2ab889b29c/src/rlai/environments/openai_gym.py#L457-L459)
with the action space to produce the final one-hot-action-context form of the state-action value function used here.

### Nonstationary Feature Scaling
All features are [scaled](https://github.com/MatthewGerber/rlai/blob/36b755098e75dd1222a802933075db2ab889b29c/src/rlai/environments/openai_gym.py#L443)
to address step-size issues when using state variables on different scales. These issues are covered nicely in an
[article](https://towardsdatascience.com/gradient-descent-the-learning-rate-and-the-importance-of-feature-scaling-6c0b416596e1)
by Daniel Godoy at Towards Data Science; however, one thing he does not cover is nonstationary state spaces. As the 
agent's policy evolves, the distribution (e.g., mean and variance) of the resulting state variables will change. If 
this distribution is used to scale the state variables (e.g., with scikit-learn's 
[`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), as Daniel suggests)
then the distribution will need to adapt over time in order to accurately reflect the system's state distribution. The
approach taken in RLAI is to extend `StandardScaler` with 
[recency-weighting](https://github.com/MatthewGerber/rlai/blob/36b755098e75dd1222a802933075db2ab889b29c/src/rlai/value_estimation/function_approximation/models/feature_extraction.py#L323).
A history of observed state values is retained, and the scaler is refit periodically with the weight on each observation
decreasing exponentially with age.