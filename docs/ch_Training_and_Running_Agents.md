# Training and Running Agents
### [rlai.runners.agent_in_environment.run](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/runners/agent_in_environment.py#L17)
```
Run an agent within an environment.

    :param args: Arguments.
```
### [rlai.runners.top_level.run](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/runners/top_level.py#L10)
```
Run RLAI.

    :param args: Arguments.
    :return: Return value of specified function.
```
### [rlai.runners.trainer.run](https://github.com/MatthewGerber/rlai/tree/master/src/rlai/runners/trainer.py#L21)
```
Train an agent in an environment.

    :param args: Arguments.
    :param thread_manager: Thread manager for the thread that is executing the current function. If None, then training
    will continue until termination criteria (e.g., number of iterations) are met. If not None, then the passed
    manager will be waited upon before starting each iteration. If the manager blocks, then another thread will need to
    clear the manager before the iteration continues. If the manager aborts, then this function will return as soon as
    possible.
    :returns: 2-tuple of the checkpoint path (if any) and the saved agent path (if any).
```
