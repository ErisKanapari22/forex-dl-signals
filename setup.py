# setup.py
from setuptools import setup, find_packages

setup(
    name="forex-dl-signals",
    version="0.1.0",
    packages=find_packages(where="."),
    python_requires=">=3.11, <3.12",
)