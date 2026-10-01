from setuptools import find_packages, setup

setup(
    name="neuromorphic-nwlamas",
    version="0.1.0",
    description="Neuromorphic nanowire network simulator for stochastic generation, topology modeling, and electrical/dynamic simulation",
    packages=find_packages(include=["neuromorphic", "neuromorphic.*"]),
    include_package_data=True,
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "networkx>=3.0",
        "matplotlib>=3.5.0",
    ],
    python_requires=">=3.8",
)
