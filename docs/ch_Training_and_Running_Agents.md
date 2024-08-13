[Home](index.md) > Training and Running Agents
### [rlai.runners.agent_in_environment.run](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/runners/agent_in_environment.py#L22)
```
Run an agent within an environment.

    :param args: Arguments.
    :return: List of run monitors.
```
### [rlai.runners.top_level.run](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/runners/top_level.py#L11)
```
Run RLAI.

    :param args: Arguments.
    :return: Return value of specified function.
```
### [rlai.runners.trainer.run](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/runners/trainer.py#L24)
```
Train an agent in an environment.

    :param args: Arguments.
    :param thread_manager: Thread manager for the thread that is executing the current function. If None, then training
    will continue until termination criteria (e.g., number of iterations) are met. If not None, then the passed
    manager will be waited upon before starting each iteration. If the manager blocks, then another thread will need to
    clear the manager before the iteration continues. If the manager aborts, then this function will return as soon as
    possible.
    :param train_function_args_callback: A callback function to be called with the arguments that will be passed to the
    training function. This gives the caller an opportunity to grab references to the internal arguments that will be
    used in training. For example, plotting from the Jupyter Lab interface grabs the state-action value estimator
    (q_S_A) from the passed dictionary to use in updating its plots. This callback is only called for fresh training. It
    is not called when resuming from a checkpoint.
    :returns: 2-tuple of the checkpoint path (if any) and the saved agent path (if any).
```
