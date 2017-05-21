"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from setuptools import find_packages, setup
import os

# import subprocess
#
# subprocess.call(["sudo", "apt-get", "install", "libgeos-dev"])

setup(name='dingo',
      version='v0.1-pre',
      author='Reiner Lemoine Institut, openego development group',
      author_email='jonathan.amme@rl-institut.de',
      description='DIstribution Network GeneratOr',
      url='https://github.com/openego/dingo',
      license='GNU GPLv3',
      packages=find_packages(),
      install_requires=['networkx >= 1.11, <= 1.11',
                        'geopy >= 1.11.0, <= 1.11.0',
                        'pandas >= 0.17.0, <= 0.19.2',
                        'pyomo >= 5.0.1, <= 5.1.1',
                        'pyproj >= 1.9.5.1, <= 1.9.5.1',
                        'sqlalchemy >= 1.0.11, <= 1.1.4',
                        'geoalchemy2 >= 0.2.6, <= 0.4.0',
                        'matplotlib  >= 1.5.3, <= 1.5.3',
                        'egoio >= 0.0.2, <= 0.0.2',
                        'oemof.db  >= 0.0.4, <= 0.0.4',
                        'egopowerflow >= 0.0.2, <= 0.0.2',
                        'shapely >= 1.5.12, <= 1.5.12',
                        'pypsa >= 0.7.1, <= 0.8.0',
			'seaborn'
                        ],
      package_data={
          'config': [
              os.path.join('config',
                           '*.cfg'),
            ],
          'data': [
              os.path.join('data',
                           '*.csv'),
            ],
          'testcases': [
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

          ]}
      )
