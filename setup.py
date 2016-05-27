#! /usr/bin/env python

from setuptools import find_packages, setup

setup(name='dingo',
      author='openego development group',
      description='DIstribution Network GeneratOr',
      packages=find_packages(),
      install_requires=['networkx >= 1.11',
                        'geopy >= 1.11.0',
                        'pandas >= 0.17.0',
                        'pyomo >= 1.9.5.1',
                        'pyproj',
                        'geoalchemy2']
     )