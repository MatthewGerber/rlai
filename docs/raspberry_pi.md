# Using RLAI with Raspberry Pi

## Introduction
The Raspberry Pi is an appealing platform for mobile computation in general and for RLAI in particular. The latest model as
of October 2021 has a 64-bit quad-core CPU with 8 GB of RAM. Add-on hardware provides a wide range of sensing and actuation
capabilities, and the entire ecosystem is quite cheap.

## Operating System
At present, the official Raspberry Pi OS is a 32-bit version of Debian running on the ARM CPU. It is possible to install most
RLAI dependencies, either directly from the package repositories of by building them from source. A specific few, particularly
JAX, are neither available in the repositories or straightforward to build from source. There is an open issue for this
[here](https://github.com/google/jax/issues/1161), and it indicates that support for 32-bit Raspberry Pi is not likely to
appear soon. I also experimented with Ubuntu Desktop 21.04 64-bit, which installs and runs without issues on the Raspberry Pi
4 Model B; however, the desktop interface is sluggish, and since this is not an LTS version it is not possible to use the
Deadsnakes repository to easily install Python 3.7 and 3.8 (the default is Python 3.9). The Raspberry Pi Imager does not provide
any other Desktop versions. Ultimately, I settled on Ubuntu Server 20.04 64-bit, which is a much slimmer OS that also installs
and runs without issues. It defaults to Python 3.8 and works fine with lighter desktop environments like XFCE. The install is
more complicated than for Desktop, but it is entirely feasible.

### Configure wireless internet

1. sudo nano /etc/wpa_supplicant.conf (edit as follows):
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
1. sudo wpa_supplicant -Dnl80211 -B iwlan0 -c/etc/wpa_supplicant.conf
1. sudo dhclient -v

### Upgrade OS
1. sudo apt update
1. sudo apt upgrade
1. sudo systemctl reboot

### Install required packages and XFCE desktop environment
1. sudo apt install gfortran python3-dev libblas-dev liblapack-dev build-essential swig python-pygame git virtualenv qt5-default xvfb
1. sudo apt install xubuntu-desktop
1. sudo systemctl reboot

## Install RLAI
1. git clone https://github.com/MatthewGerber/rlai.git
1. cd rlai
1. virtualenv -p python3.8 venv
1. . venv/bin/activate
1. pip install -U pip setuptools wheel
1. Box2D
  1. git clone https://github.com/pybox2d/pybox2d
  1. cd pybox2d
  1. python setup.py build
  1. python setup.py install
1. PyQt5
  1. pip install PyQt-builder
  1. wget https://files.pythonhosted.org/packages/8e/a4/d5e4bf99dd50134c88b95e926d7b81aad2473b47fde5e3e4eac2c69a8942/PyQt5-5.15.4.tar.gz
  1. tar -xvzf PyQt5-5.15.4.tar.gz
  1. cd PyQt5-5.15.4/
  1. sip-install
1. JAX
  1. wget https://files.pythonhosted.org/packages/0f/85/0499931fe8e9dc05f4dd5ef989be2db4653d429adf08b9371fc259402af0/jax-0.2.21.tar.gz
  1. tar -xvzf jax-0.2.21.tar.gz
  1. cd jax-0.2.21
  1. pip install numpy==1.21.2 six wheel
  1. cd jax
  1. python build/build.py
  1. pip install dist/*.whl
  1. pip install .
  1. edit setup.py to change jax[cpu]==0.2.17 to jax==0.2.17 (the cpu extra is not needed because we just installed jaxlib)
1. Install and test RLAI
  1. pip install -e .[dev]
  1. pytest ./test (or, if running without a display:  HEADLESS=True pytest ./test)

## IDE:  PyCharm Community Edition

## References
1. https://linuxhint.com/install-ubuntu-desktop-20-04-lts-on-raspberry-pi-4
1. https://jax.readthedocs.io/en/latest/developer.html