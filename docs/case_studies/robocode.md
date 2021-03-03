![robocode](./robocode.png)

# Robocode
Robocode is a programming game in which robot agents are programmed to operate in a simulated battle environment. Full
documentation about the game can be found [here](https://robowiki.net/wiki/Main_Page). Robocode has several features
that make it an appealing testbed for reinforcement learning:

* Rich dynamics:  The state space is continuous, and each robot has several dimensions of interrelated action, some of
  which are discrete (fire or not) and some of which are continuous (robot, gun, and radar rotation). There are many 
  options for reward functions. The [official scoring function](https://robowiki.net/wiki/Robocode/Scoring) is a mixture 
  of many variables, and simpler reward functions can be designed (e.g., robot energy level lost or gained).
* Multi-agent with teaming:  Robocode is inherently multi-agent. Teams can be formed, and team members can communicate
  by passing messages to each other. Perhaps a communication language might be learnable?
* Simple integration with `rlai`:  The architecture of Robocode lends itself to a simple integration with `rlai` via
  local network-based exchange of state and action information.
  
The purpose of this case study is to explore the use of `rlai` for learning robot policies.

## Installing Robocode

1. Download the Robocode installer 
   [here](https://github.com/MatthewGerber/robocode/raw/master/build/robocode-rlai-setup.jar). Note that this is a 
   customized build of Robocode containing a few tweaks to make it compatible with `rlai`. It provides robots with 
   elevated permissions, particularly those related to TCP (socket) communication with the localhost (127.0.0.1) and
   reflection. These permissions are needed to communicate with the `rlai` server, and in general they do not pose much
   of a security risk; however, it is probably a good idea to avoid importing other robots into this installation of
   Robocode.
1. Run the Robocode installer. Install to a directory such as `robocode_rlai`.

## Training a Robocode Agent

1. Start the Robocode `rlai` environment. This is most easily done using the 
   [JupyterLab notebook](../jupyterlab_guide.md). Note that there is already a configuration saved in the notebook that 
   should suffice as a simple demonstration of reinforcement learning with Robocode.
1. Start Robocode from the directory into which you installed it above. Add a few robots as well as the `RlaiRobot`, 
   then begin the battle. If this is successful, then you will see the `RlaiRobot` moving, firing, etc. This is the 
   start of training, so the agent will likely appear random for quite some time until its policy develops.
   
As noted above, this is just a simple demonstration, and the learned robot will probably not perform very well. Below
are some things to consider:

* What should the reward signal be?
* What should the state features be?
* How should the learning model (e.g., stochastic gradient descent) be parameterized?

The first two points above are addressed in 
[robocode.py](https://github.com/MatthewGerber/rlai/blob/master/src/rlai/environments/robocode.py). The third question
is addressed in the command-line (or JupyterLab) arguments (e.g., loss function, learning rate, etc.). Feel free to 
change all of these. In order to restart training, you will need to first restart the Robocode application and then
restart the `rlai` environment by killing the command (if using the CLI) or by restarting the JupyterLab kernel. This
is a bit tedious but is required to reset the state of each.
