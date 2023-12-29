[Home](index.md) > Using RLAI with Raspberry Pi
* Content
{:toc}

See [here](https://matthewgerber.github.io/raspberry-py/#ubuntu-operating-system) for instructions on configuring 
Ubuntu on Raspberry Pi.

# Install RLAI

## Configure Virtual Environment
1. `git clone https://github.com/MatthewGerber/rlai.git`
2. `cd rlai`
3. `virtualenv -p python3.8 venv`
4. `. venv/bin/activate`
5. `pip install -U pip setuptools wheel`

## Box2D
1. `git clone https://github.com/pybox2d/pybox2d`
2. `cd pybox2d`
3. `python setup.py build`
4. `python setup.py install`

## PyQt5
1. `pip install PyQt-builder`
2. `wget https://files.pythonhosted.org/packages/8e/a4/d5e4bf99dd50134c88b95e926d7b81aad2473b47fde5e3e4eac2c69a8942/PyQt5-5.15.4.tar.gz`
3. `tar -xvzf PyQt5-5.15.4.tar.gz`
4. `cd PyQt5-5.15.4`
5. `sip-install`

## JAX
1. `wget https://files.pythonhosted.org/packages/0f/85/0499931fe8e9dc05f4dd5ef989be2db4653d429adf08b9371fc259402af0/jax-0.2.21.tar.gz`
2. `tar -xvzf jax-0.2.21.tar.gz`
3. `cd jax-0.2.21`
4. `pip install numpy==1.21.2 six wheel`
5. `cd jax`
6. `python build/build.py`
7. `pip install dist/*.whl`
8. `pip install .`

## Install and Test RLAI
1. `pip install -e .[dev]`
2. `pytest ./test` (or, if running without a display:  `HEADLESS=True pytest ./test`)

## VNC (Remote Desktop)
Using the above configuration, I found the following VNC setup works best, providing automatic screen scaling and 
reliable operation.
1. Install VNC server:  `sudo apt install tigervnc-standalone-server`
2. Start VNC server:  `vncserver :1 -localhost yes`
3. Start SSH tunnel from client, where `XXXX` is the username and `YYYY` is the IP address:  `ssh -L 59000:localhost:5901 -C -l XXXX YYYY`
4. Use the [RealVNC Viewer][https://www.realvnc.com/en/connect/download/viewer/macos] to connect to `localhost:59000`.

## References
1. https://linuxhint.com/install-ubuntu-desktop-20-04-lts-on-raspberry-pi-4
2. https://jax.readthedocs.io/en/latest/developer.html
