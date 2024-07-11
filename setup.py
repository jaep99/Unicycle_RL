from setuptools import setup, find_packages

setup(
    name='custom_gym',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'gymnasium',
        'mujoco',
        'stable-baselines3[extra]'
    ],
    python_requires='==3.11.9',
)


# Gymansium version 0.29.1 works