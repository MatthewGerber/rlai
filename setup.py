from setuptools import (
    find_namespace_packages,
    setup
)

INSTALL_REQUIREMENTS = [
    'scipy==1.5.2',
    'matplotlib==3.3.1',
    'numpy==1.19.0'
]

TEST_REQUIREMENTS = [
    'pytest==5.3',
    'pytest-cov==2.8',
    'pytest-runner==5.2',
    'nose==1.3.7'
]

DEV_REQUIREMENTS = [
    'bump2version==1.0',
    'flake8==3.7'
]

setup(
    name='rl',
    version='0.1.0.dev0',
    description='Reinforcement Learning',
    author='Matthew Gerber',
    author_email='gerber.matthew@gmail.com',
    url='https://github.com/MatthewGerber/rl',
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
