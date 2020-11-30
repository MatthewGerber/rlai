from os import path

from setuptools import (
    setup, find_namespace_packages
)

INSTALL_REQUIREMENTS = [
    'scipy==1.5.2',
    'matplotlib==3.3.1',
    'numpy==1.19.0',
    'gym==0.17.3',
    'mujoco-py==2.0.2.13'
]

TEST_REQUIREMENTS = [
    'pytest==5.3',
    'pytest-cov==2.8',
    'pytest-runner==5.2',
    'nose==1.3.7'
]

DEV_REQUIREMENTS = [
    'bump2version==1.0.1',
    'flake8==3.7',
    'md-toc==7.0.3'
]

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rlai',
    version='0.11.0',
    description='Reinforcement Learning:  An Introduction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matthew Gerber',
    author_email='gerber.matthew@gmail.com',
    url='https://github.com/MatthewGerber/rlai',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='==3.8.5',
    install_requires=[
        INSTALL_REQUIREMENTS
    ],
    tests_require=TEST_REQUIREMENTS,
    extras_require={
        'test:python_version == "3.8"': TEST_REQUIREMENTS,
        'dev:python_version == "3.8"': TEST_REQUIREMENTS + DEV_REQUIREMENTS
    }
)
