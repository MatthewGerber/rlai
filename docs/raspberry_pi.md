[Home](index.md) > Using RLAI with Raspberry Pi
* Content
{:toc}

See [here](https://matthewgerber.github.io/raspberry-py/#ubuntu-operating-system) for instructions on configuring 
Ubuntu on Raspberry Pi.

# Prerequisites

## Configure Virtual Environment
1. `git clone https://github.com/MatthewGerber/rlai.git`
2. `cd rlai`
3. `python3.11 -m venv venv'
4. `. venv/bin/activate`
5. `pip install -U pip`

## PyQt6

1. `sudo apt install build-essential libgles2 libdrm-dev`
2. `wget https://download.qt.io/official_releases/qt/6.6/6.6.1/single/qt-everywhere-src-6.6.1.tar.xz`

1. `sudo apt install qt6-base-dev qtchooser`
2. `qtchooser -install qt6 $(which qmake6)`
3. `export QT_SELECT=qt6`

OLD

1. `pip install PyQt-builder`
2. `wget https://files.pythonhosted.org/packages/8e/a4/d5e4bf99dd50134c88b95e926d7b81aad2473b47fde5e3e4eac2c69a8942/PyQt5-5.15.4.tar.gz`
3. `tar -xvzf PyQt5-5.15.4.tar.gz`
4. `cd PyQt5-5.15.4`
5. `sip-install`

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
