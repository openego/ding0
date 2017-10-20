#!/usr/bin/env python3

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


# ===== IMPORTS AND CONFIGURATION =====

# import DB interface
from ding0.tools import db

# import required modules of DING0
from ding0.core import NetworkDing0
from ding0.tools.logger import setup_logger
from ding0.tools.results import save_nd_to_pickle
import logging

# define logger
logger = setup_logger()

# ===== MAIN =====

# database connection
conn = db.connection(section='oedb')

# instantiate new ding0 network object
nd = NetworkDing0(name='network')

# choose MV Grid Districts to import
mv_grid_districts = [3545]

# run DING0 on selected MV Grid District
nd.run_ding0(conn=conn,
             mv_grid_districts_no=mv_grid_districts)

# export grids to database
# nd.export_mv_grid(conn, mv_grid_districts)
# nd.export_mv_grid_new(conn, mv_grid_districts)

conn.close()

# export grid to file (pickle)
save_nd_to_pickle(nd, filename='ding0_grids_example.pkl')
