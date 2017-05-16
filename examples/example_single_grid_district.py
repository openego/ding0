#!/usr/bin/env python3

"""This is a simple example file for DINGO.

__copyright__ = "Reiner Lemoine Institut, openego development group"
__license__ = "GNU GPLv3"
__author__ = "Jonathan Amme, Guido Pleßmann"
"""

# ===== IMPORTS AND CONFIGURATION =====

# import DB interface from oemof
import oemof.db as db

# import required modules of DINGO
from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo, results
from dingo.tools.logger import setup_logger

# define logger
logger = setup_logger()

# load parameters from configs
cfg_dingo.load_config('config_db_tables.cfg')
cfg_dingo.load_config('config_calc.cfg')
cfg_dingo.load_config('config_files.cfg')
cfg_dingo.load_config('config_misc.cfg')

# ===== MAIN =====

# database connection
conn = db.connection(section='oedb')

# instantiate new dingo network object
nd = NetworkDingo(name='network')

# choose MV Grid Districts to import
mv_grid_districts = [3545]

