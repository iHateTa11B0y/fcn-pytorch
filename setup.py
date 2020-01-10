from setuptools import setup, find_packages

requirements = ['torch', 'cvtorch']

setup(
    name="fcn_benchmark",
    version="0.0.1",
    author="wenhao",
    url="https://github.com/iHateTa11B0y/fcns.git",
    description="fcn in pytorch",
    packages=find_packages(exclude=("configs", "tools",)),
    # install_requires=requirements,
)

