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
                        'geoalchemy2',
                        'matplotlib',
                        'ego.io',# >= 0.0.1rc7',
                        'oemof.db',
                        # 'ego.powerflow',
                        ]
      # dependency_links=['https://github.com/openego/ego.io/archive/856769eb79e5342c349fe479e8c42da6481e122c.zip#egg=ego.io-0.0.1rc7+git.856769e']
     )
