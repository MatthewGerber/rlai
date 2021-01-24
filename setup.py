from os import path

from setuptools import (
    setup, find_namespace_packages
)

INSTALL_REQUIREMENTS = [
    'scipy==1.5.2',
    'matplotlib==3.3.1',
    'numpy==1.19.0',
    'gym==0.17.3',
    'Box2D==2.3.10',
    'python-dateutil==2.8.1',
    'importlib-metadata==3.1.1',
    'packaging==20.7',
    'more-itertools==8.6.0',
    'attrs==20.3.0',
    'pyparsing==2.4.7',
    'future==0.18.2',
    'scikit-learn==0.24',
    'pandas==1.1.5',
    'patsy==0.5.1'
]

TEST_REQUIREMENTS = [
    'pytest==5.3',
    'pytest-cov==2.8',
    'coverage==5.3',
    'pytest-runner==5.2',
    'nose==1.3.7',
    'flake8==3.7',
    'coveralls==2.2.0'
]

DEV_REQUIREMENTS = [
    'bump2version==1.0.1'
]

MUJOCO_REQUIREMENTS = [
    'mujoco-py==2.0.2.13'
]

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rlai',
    version='0.15.0',
    description='Reinforcement Learning:  An Introduction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matthew Gerber',
    author_email='gerber.matthew@gmail.com',
    url='https://matthewgerber.github.io/rlai',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='==3.8.5',
    install_requires=[
        INSTALL_REQUIREMENTS
    ],
    tests_require=TEST_REQUIREMENTS,
    extras_require={
        'test:': TEST_REQUIREMENTS,
        'dev:': TEST_REQUIREMENTS + DEV_REQUIREMENTS,
        'mujoco:': MUJOCO_REQUIREMENTS
    },
    entry_points={
        'console_scripts': [
            'rlai=rlai.runners.top_level:run'
        ]
    }
)
