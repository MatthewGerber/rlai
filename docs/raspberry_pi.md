[Home](index.md) > Using RLAI with Raspberry Pi
* Content
{:toc}

# Using RLAI with Raspberry Pi
Most of the RLAI functionality can run on a Raspberry Pi. This opens possibilities for building physical systems with 
the Pi's GPIO ports and then learning control policies for these systems with RLAI. See 
[here](https://matthewgerber.github.io/raspberry-py/#operating-system) for instructions on configuring an operating
system on the Raspberry Pi. This page provides details on installing RLAI on the Pi for use within a Python project 
running the physical system (e.g., the cart-pole system described [here](https://matthewgerber.github.io/cart-pole)).

# Installing RLAI 
1. Log in to the Pi via the desktop or SSH connection.
2. Install dependencies:
   ```shell
   sudo apt install build-essential swig
   ```
3. Configure a virtual environment within the project:
   ```shell
   cd /path/to/rpi/project
   python3 -m venv venv
   . venv/bin/activate
   pip install -U pip
   ```
4. Clone RLAI and install it within the project's virtual environment:
   ```shell
   cd ~
   git clone https://github.com/MatthewGerber/rlai.git
   pip install rlai
   ```
   This step will take quite a while, since pip will need to compile several of the RLAI dependencies from source.

# Limitations
1. RLAI depends on Qt6 for graphical rendering of certain simulations and plots. These renderings will not be possible
   on the Pi.