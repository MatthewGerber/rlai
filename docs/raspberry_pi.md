[Home](index.md) > Using RLAI with Raspberry Pi
* Content
{:toc}

# Using RLAI with Raspberry Pi
Most of the RLAI functionality can run on a Raspberry Pi. This opens possibilities for building physical systems with 
the Pi's GPIO ports and then learning control policies for these systems with RLAI. See 
[here](https://matthewgerber.github.io/raspberry-py/#operating-system) for instructions on configuring an operating
system on the Raspberry Pi. Installing and using RLAI on the Pi is similar to using it elsewhere:
1. Log in to the Pi via the desktop or SSH connection.
2. Install build dependencies for RLAI:
   ```shell
   sudo apt install build-essential swig
   ```
3. Add RLAI as a dependency to the project on the Pi, either from [PyPI](https://pypi.org/project/rlai/) (e.g., 
with `poetry add rlai`) or as a submodule as shown in the cart-pole system described 
[here](https://matthewgerber.github.io/cart-pole).
4. Install the project.
Note that RLAI depends on Qt6 for graphical rendering of certain simulations and plots. These renderings will not be 
possible on the Pi, since Qt6 isn't installed.
5. If poetry installations hang, you might require the following:
   ```shell
   export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
   ```
