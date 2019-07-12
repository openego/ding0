# coding: utf-8
"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "jh-RLI"

import os
from egoio.tools import db
from ding0.io.exporter_log import pickle_export_logger
from ding0.tools.results import load_nd_from_pickle
from ding0.io.export import export_network
from ding0.io.db_export import METADATA, create_ding0_sql_tables, \
    export_all_pkl_to_db, db_tables_change_owner, drop_ding0_db_tables
from sqlalchemy.orm import sessionmaker

##################################
# LOG_FILE_PATH = 'pickle_log'
LOG_FILE_PATH = os.path.join(os.path.expanduser("~"), '.ding0_log', 'pickle_log')
pickle_export_logger(LOG_FILE_PATH)
###################################

# database connection/ session
oedb_engine = db.connection(section='oedb')
session = sessionmaker(bind=oedb_engine)()

SCHEMA = "model_draft"

create_ding0_sql_tables(oedb_engine, SCHEMA)
db_tables_change_owner(oedb_engine, SCHEMA)
# drop_ding0_db_tables(oedb_engine)

# pickle file locations path to RLI_Daten_Flex01 mount
pkl_filepath = "/home/local/RL-INSTITUT/jonas.huber/rli/Daten_flexibel_01/Ding0/20180823154014"


# choose MV Grid Districts to import use list of integers
# f. e.: multiple grids = list(range(1, 3609))
grids = [1658]

# generate all the grids and push them to oedb
for grid_no in grids:

    try:
        nw = load_nd_from_pickle(os.path.join(pkl_filepath, 'ding0_grids__{}.pkl'.format(grid_no)))
    except:
        print('Something went wrong, created log entry in: {}'.format(LOG_FILE_PATH))
        with open(LOG_FILE_PATH, 'a') as log:
            log.write('ding0_grids__{}.pkl not present to the current directory\n'.format(grid_no))
            pass

        continue

    # Extract data from network and put it to DataFrames for csv and for oedb
    # run_id is manually provided -> folder name or nw.metadata['run_id'] provide the run_id value
    network = export_network(nw, run_id=20180823154014)

    # set SRID form pickle file
    PICKLE_SRID = int(nw.config['geo']['srid'])


    # provide run id for pickle upload

    export_all_pkl_to_db(oedb_engine, SCHEMA, network, PICKLE_SRID, grid_no)


# db_tables_change_owner(oedb_engine, SCHEMA)
