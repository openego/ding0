#!/usr/bin/env python3

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


# ===== IMPORTS AND CONFIGURATION =====

# import DB interface
from dingo.tools import db

# import required modules of DINGO
from dingo.core import NetworkDingo
from dingo.tools.logger import setup_logger
from dingo.tools.results import save_nd_to_pickle
import logging

# define logger
logger = setup_logger()

# ===== MAIN =====

# database connection
conn = db.connection(section='oedb')

# instantiate new dingo network object
nd = NetworkDingo(name='network')

# choose MV Grid Districts to import
mv_grid_districts = [3545]

# run DINGO on selected MV Grid District
nd.run_dingo(conn=conn,
             mv_grid_districts_no=mv_grid_districts)

# export grids to database
nd.control_circuit_breakers(mode='close')
# nd.export_mv_grid(conn, mv_grid_districts)
# nd.export_mv_grid_new(conn, mv_grid_districts)

conn.close()

# export grid to file (pickle)
save_nd_to_pickle(nd, filename='dingo_grids_example.pkl')
