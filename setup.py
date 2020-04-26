from setuptools import setup, find_packages

setup(name='generalization',
      version='0.0.1',
      install_requires=['gym', 'stable-baselines[mpi]'],
      packages=find_packages()
)