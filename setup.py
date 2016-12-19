#! /usr/bin/env python

from setuptools import find_packages, setup

setup(name='dingo',
      author='openego development group',
      description='DIstribution Network GeneratOr',
      packages=find_packages(),
      install_requires=['networkx >= 1.11, <= 1.11',
                        'geopy >= 1.11.0, <= 1.11.0',
                        'pandas >= 0.17.0 <= 0.19.1',
                        'pyomo >= 5.0.1, <= 5.0.1',
                        'pyproj >= 1.9.5.1, >= 1.9.5.1',
                        'geoalchemy2 >= 0.3.0, <=0.4.0',
                        'matplotlib  >=1.5.3, <= 1.5.3',
                        'egoio >=0.0.2, <= 0.0.2',
                        'oemof.db  >=0.0.4, <= 0.0.4',
                        'egopowerflow >=0.0.2, <= 0.0.2',
                        ]
     )
