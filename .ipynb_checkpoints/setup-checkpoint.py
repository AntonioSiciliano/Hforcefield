#import setuptools
from __future__ import print_function
from numpy.distutils.core import setup, Extension

setup(name = "Hmodel",
      version = "0.1",
      description = "Toy model calculator",
      author = "Antonio Siciliano",
      packages = ["Hmodel"],
      package_dir = {"Hmodel": "Modules"},
      license = "GPLv3")


def readme():
    with open("README.md") as f:
        return f.read()
