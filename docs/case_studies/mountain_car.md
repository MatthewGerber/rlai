[Home](../index.md) > Mountain Car
* Content
{:toc}
  
# Introduction
The mountain car is not sufficiently powerful to climb the hill directly, but must instead develop a strategy based on 
the surrounding slopes. You can read more about this environment 
[here](https://gymnasium.farama.org/environments/classic_control/mountain_car/). Below is an example of running a random 
(untrained) agent in this environment. The episode takes quite a long time to terminate.

{% include youtubePlayer.html id="fnB_84YAW08" %}

# Training
Train a control agent for the mountain car environment with the following command.
```
rlai train --agent "rlai.gpi.state_action_value.ActionValueMdpAgent" --continuous-state-discretization-resolution 0.005 --gamma 0.95 --environment "rlai.core.environments.gymnasium.Gym" --gym-id "MountainCar-v0" --render-every-nth-episode 1000 --video-directory "~/Desktop/mountaincar_videos" --train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi" --mode "Q_LEARNING" --num-improvements 10000 --num-episodes-per-improvement 1 --epsilon 0.01 --make-final-policy-greedy True --num-improvements-per-plot 100 --num-improvements-per-checkpoint 100 --checkpoint-path "~/Desktop/mountaincar_checkpoint.pickle" --save-agent-path "~/Desktop/mountaincar_agent.pickle"
```

Arguments are explained below.
* `train`:  Train the agent.
* `--agent "rlai.gpi.state_action_value.ActionValueMdpAgent"`:  Standard action-value MDP agent. 
* `--continuous-state-discretization-resolution 0.005`:  Discretize the continuous state space into discrete intervals 
  with resolution 0.005. The methods used here are for discrete-state problems, so some type of discretization of the 
  continuous state space is required.
* `--gamma 0.95`:  Discount the reward passed backward to previous state-action pairs. This is an important part of 
  getting the agent to learn a useful policy. Without discounting, particularly in early iterations, the protracted
  fumbling about with poor actions would eventually be rewarded fully for a subsequent success. By discounting prior 
  actions, the agent eventually learns to focus on later actions that are instrumental to success.
* `--environment "rlai.core.environments.gymnasium.Gym"`:  Environment class. 
* `--gym-id "MountainCar-v0"`:  Gym environment identifier.
* `--render-every-nth-episode 1000`:  Render a video every 1000 episodes (1000 improvements).
* `--video-directory "~/Desktop/mountaincar_videos"`:  Where to store rendered videos.
* `--train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi"`:  Run iterative temporal-differencing 
  on the agent's state-action value function. 
* `--mode "Q_LEARNING"`:  Use q-learning to bootstrap the value of the next state-action pair. 
* `--num-improvements 10000`:  Number of policy improvements (iterations).
* `--num-episodes-per-improvement 1`:  Number of episodes per improvement.
* `--epsilon 0.01`:  Probability of behaving randomly at each time step.
* `--make-final-policy-greedy True`:  After all learning iterations, make the final policy greedy (i.e., `epsilon=0.0`).
* `--num-improvements-per-plot 100`:  Plot training performance every 100 iterations.
* `--num-improvements-per-checkpoint 100`:  Checkpoint the learning process every 100 iterations, enabling resumption.
* `--checkpoint-path "~/Desktop/mountaincar_checkpoint.pickle"`:  Where to store the checkpoint.
* `--save-agent-path "~/Desktop/mountaincar_agent.pickle"`:  Where to save the final agent.

Note that, unlike other tasks such as the [inverted pendulum](./inverted_pendulum.md), no value is passed for `--T` 
(maximum number of time steps per episode). This is because there is no way to predict how long an episode will last, 
particularly episodes earlier in the training. All episodes must be permitted to run until success in order to learn 
a useful policy. The training progression is shown below.

![acrobot](https://github.com/MatthewGerber/rlai/raw/master/trained_agents/mountaincar/mountaincar_training.png)

In the left sub-figure above, the left y-axis shows the negation of time taken to reach the goal, the right y-axis shows 
the size of the state space, and the x-axis shows improvement iterations for the agent's policy. The right sub-figure 
shows the same reward y-axis but along a time x-axis. Based on the learning trajectory, it appears that little 
subsequent improvement would be gained were the agent to continue improving its policy; as shown below, the results are 
quite satisfactory after 30 minutes of wallclock training time.

# Results
The video below shows the trained agent controlling the car. Note how the agent develops an oscillating movement.

{% include youtubePlayer.html id="qjVdoYYnriA" %}
