#! /usr/bin/env python

from setuptools import find_packages, setup
import subprocess

subprocess.call(["sudo", "apt-get", "install", "libgeos-dev"])

setup(name='dingo',
      author='openego development group',
      description='DIstribution Network GeneratOr',
      packages=find_packages(),
      install_requires=['networkx >= 1.11',
                        'geopy >= 1.11.0',
                        'pandas >= 0.17.0',
                        'pyomo >= 1.9.5.1',
                        'pyproj',
                        'geoalchemy2',
                        #'matplotlib', #should be included but fails via pip3
                        'ego.io >= 0.0.1-pre',
                        'oemof.db']
     )

