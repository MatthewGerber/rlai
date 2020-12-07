# Inverted Pendulum
The inverted pendulum is also known as cart-pole balancing.

## Training

To train a control agent for the inverted pendulum environment, first install the `rlai` package and then run the 
following command.
```
rlai_train --train --agent "rlai.agents.mdp.StochasticMdpAgent" --continuous-state-discretization-resolution 0.1 --gamma 1 --environment "rlai.environments.openai_gym.Gym" --gym-id "CartPole-v1" --render-every-nth-episode 5000 --video-directory "~/Desktop/cartpole_videos" --train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi" --mode "Q_LEARNING" --num-improvements 5000 --num-episodes-per-improvement 50 --T 1000 --epsilon 0.01 --make-final-policy-greedy True --num-improvements-per-plot 100 --num-improvements-per-checkpoint 100 --checkpoint-path "~/Desktop/cartpole_checkpoint.pickle" --save-agent-path "~/Desktop/cartpole_agent.pickle"
```

Arguments are explained below.
```
--train 
--agent "rlai.agents.mdp.StochasticMdpAgent" 
--continuous-state-discretization-resolution 0.1 
--gamma 1 
--environment "rlai.environments.openai_gym.Gym" 
--gym-id "CartPole-v1" 
--render-every-nth-episode 5000 
--video-directory "~/Desktop/cartpole_videos" 
--train-function "rlai.gpi.temporal_difference.iteration.iterate_value_q_pi" 
--mode "Q_LEARNING" 
--num-improvements 5000 
--num-episodes-per-improvement 50 
--T 1000 
--epsilon 0.01 
--make-final-policy-greedy True 
--num-improvements-per-plot 100 
--num-improvements-per-checkpoint 100 
--checkpoint-path "~/Desktop/cartpole_checkpoint.pickle" 
--save-agent-path "~/Desktop/cartpole_agent.pickle"
```

## Results

![inverted-pendulum](https://github.com/MatthewGerber/rlai/blob/master/trained_agents/cartpole/cartpole_training.png)
