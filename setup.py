"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from setuptools import find_packages, setup
import os

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='ding0',
      version='v0.1.11dev',
      author='Reiner Lemoine Institut, openego development group',
      author_email='jonathan.amme@rl-institut.de',
      description='DIstribution Network GeneratOr',
      url='https://github.com/openego/ding0',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=requirements,
      package_data={
          'ding0': [
              os.path.join('config',
                           '*.cfg'),
              os.path.join('data',
                           '*.csv'),
              os.path.join('grid',
                           'mv_grid',
                           'tests',
                           'testcases',
                           '*.vrp'),
              os.path.join('grid',
                           'mv_grid',
                           'tests',
                           'testcases',
                           'Augerat',
                           '*.vrp'),
              os.path.join('grid',
                           'mv_grid',
                           'tests',
                           'testcases',
                           'Augerat-tcc',
                           '*.vrp'),
              os.path.join('grid',
                           'mv_grid',
                           'tests',
                           'testcases',
                           'Takes-tcc',
                           '*.vrp'),
              os.path.join('grid',
                           'mv_grid',
                           'tests',
                           'testcases',
                           'Vigo',
                           '*.vrp'),

          ]},
      classifiers=[
        "Development Status :: 3 - Alpha",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering"],
      )
