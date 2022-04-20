[Home](index.md)

# Using RLAI with Raspberry Pi
* Content
{:toc}

## Introduction
The Raspberry Pi is an appealing platform for mobile computation in general and for RLAI in particular. The latest model
as of October 2021 has a 64-bit quad-core CPU with 8 GB of RAM. Add-on hardware provides a wide range of sensing and 
actuation capabilities, and the entire ecosystem is quite affordable.

## Operating System
At present, the official Raspberry Pi OS is a 32-bit version of Debian running on the 64-bit ARM CPU. Thus, the OS 
presents a 32-bit CPU to all software in the OS. It is possible to install most RLAI dependencies, either directly from 
the package repositories of by building them from source. A specific few, particularly JAX, are neither available in the 
repositories nor straightforward to build from source for the ARM CPU. There is an open issue for this 
[here](https://github.com/google/jax/issues/1161), and it indicates that support for 32-bit Raspberry Pi is not likely 
to appear soon. I experimented with Ubuntu Desktop 21.04 64-bit, which installs and runs without issues on the Raspberry 
Pi 4 Model B; however, the desktop interface is sluggish, and since this is not an LTS version it is not possible to use 
the Deadsnakes repository to install Python 3.7 and 3.8 (the Ubuntu default is Python 3.9). The Raspberry Pi Imager does 
not provide any other Ubuntu Desktop versions. Ultimately, I settled on Ubuntu Server 20.04 64-bit, which is a much 
slimmer OS that also installs and runs without issues. It defaults to Python 3.8 and works fine with lighter desktop 
environments like XFCE. The installation is more complicated than for Ubuntu Desktop, but it is entirely feasible. 
Detailed instructions are provided below.

### Image the Raspberry Pi SD Card
1. Install and start the Raspberry Pi Imager.
2. Select Ubuntu Server 20.04 64-bit within the Raspberry Pi Imager, then write the OS to the SD card.
3. Insert the SD card into the Raspberry Pi and boot.

### Configure Wireless Internet

1. `sudo nano /etc/wpa_supplicant.conf` (edit as follows, replacing values as indicated):
```
country=US
ctrl_interface=DIR=/var/run/wpa_supplicant
update_config=1
network={
  ssid="Your Wi-Fi SSID"
  scan_ssid=1
  psk="Your Wi-Fi Password"
  key_mgmt=WPA-PSK
}
```
2. Enable wireless interface:  `sudo wpa_supplicant -Dnl80211 -B iwlan0 -c/etc/wpa_supplicant.conf`
2. Obtain wireless address:  `sudo dhclient -v`

### Upgrade OS
1. `sudo apt update`
1. `sudo apt upgrade`
1. `sudo systemctl reboot`

### Install Required Packages and XFCE Desktop Environment
1. `sudo apt install gfortran python3-dev libblas-dev liblapack-dev build-essential swig python-pygame git virtualenv qt5-default xvfb ffmpeg`
1. `sudo apt install xubuntu-desktop`
1. `sudo systemctl reboot`

## Install RLAI

### Configure Virtual Environment
1. `git clone https://github.com/MatthewGerber/rlai.git`
2. `cd rlai`
3. `virtualenv -p python3.8 venv`
4. `. venv/bin/activate`
5. `pip install -U pip setuptools wheel`

### Box2D
1. `git clone https://github.com/pybox2d/pybox2d`
2. `cd pybox2d`
3. `python setup.py build`
4. `python setup.py install`

### PyQt5
1. `pip install PyQt-builder`
2. `wget https://files.pythonhosted.org/packages/8e/a4/d5e4bf99dd50134c88b95e926d7b81aad2473b47fde5e3e4eac2c69a8942/PyQt5-5.15.4.tar.gz`
3. `tar -xvzf PyQt5-5.15.4.tar.gz`
4. `cd PyQt5-5.15.4`
5. `sip-install`

### JAX
1. `wget https://files.pythonhosted.org/packages/0f/85/0499931fe8e9dc05f4dd5ef989be2db4653d429adf08b9371fc259402af0/jax-0.2.21.tar.gz`
2. `tar -xvzf jax-0.2.21.tar.gz`
3. `cd jax-0.2.21`
4. `pip install numpy==1.21.2 six wheel`
5. `cd jax`
6. `python build/build.py`
7. `pip install dist/*.whl`
8. `pip install .`

### Install and Test RLAI
1. `pip install -e .[dev]`
2. `pytest ./test` (or, if running without a display:  `HEADLESS=True pytest ./test`)

## IDE:  PyCharm Community Edition

## VNC (Remote Desktop)
Using the above configuration, I found the following VNC setup works best, providing automatic screen scaling and 
reliable operation.
1. Install VNC server:  `sudo apt install tigervnc-standalone-server`
2. Start VNC server:  `LD_PRELOAD=/lib/aarch64-linux-gnu/libgcc_s.so.1 vncserver :1 -localhost no`
3. Start SSH tunnel from client, where `XXXX` is the user name and `YYYY` is the IP address:  `ssh -L 59000:localhost:5901 -C -l XXXX YYYY`
4. Use the TigerVNC client to connect to `localhost:59000`.

## References
1. https://linuxhint.com/install-ubuntu-desktop-20-04-lts-on-raspberry-pi-4
2. https://jax.readthedocs.io/en/latest/developer.html
