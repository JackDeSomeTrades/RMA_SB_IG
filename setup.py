from setuptools import setup

setup(
    name="rma_sb_ig",
    version="0.1.0",
    packages=["rma_sb_ig"],
    install_requires=["isaacgym", "stable-baselines3", "torch", "numpy"],
)
