from os import path

from setuptools import (
    setup, find_namespace_packages
)

INSTALL_REQUIREMENTS = [

    # core requirements
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
    'patsy==0.5.1',
    'jax[cpu]==0.2.17',
    'pyqtgraph==0.12.2',
    'PyQt5==5.15.4',
    'tabulate==0.8.9',

    # jupyter requirements
    'jupyterlab==3.0.6',
    'ipython==7.19.0',
    'ipympl==0.6.3',
    'tornado==6.1.0'
]

TEST_REQUIREMENTS = [
    'pytest==6.2.4',
    'pytest-cov==2.12.1',
    'coverage==5.5',
    'pytest-runner==5.3.1',
    'nose==1.3.7',
    'flake8==3.9.2',
    'coveralls==3.2.0',
    'xvfbwrapper==0.2.9'
]

DEV_REQUIREMENTS = [
    'bump2version==1.0.1'
]

# mujoco requires a more elaborate registration and installation process, and the bindings below only work after going
# through the registration and install.
MUJOCO_REQUIREMENTS = [
    'mujoco-py==2.0.2.13'
]

with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='rlai',
    version='0.20.0',
    description='Reinforcement Learning:  An Introduction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Matthew Gerber',
    author_email='gerber.matthew@gmail.com',
    url='https://matthewgerber.github.io/rlai',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='~=3.7',
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
