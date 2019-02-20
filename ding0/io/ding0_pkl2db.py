# coding: utf-8

import os

import numpy as np
import json

from egoio.tools import db

# import required modules of DING0
from ding0.tools.logger import setup_logger
from ding0.tools.results import load_nd_from_pickle
from ding0.io.export import export_network
from ding0.io.db_export import METADATA, create_ding0_sql_tables, \
    export_all_pkl_to_db, db_tables_change_owner, drop_ding0_db_tables
from sqlalchemy.orm import sessionmaker

# ToDo: Create logger as function
##################################
# LOG_FILE_PATH = 'pickle_log'
LOG_FILE_PATH = os.path.join(os.path.expanduser("~"), '.ding0_log', 'pickle_log')

# does the file exist?
if not os.path.isfile(LOG_FILE_PATH):
    print('ding0 log-file {file} not found. '
          'This might be the first run of the tool. '
          .format(file=LOG_FILE_PATH))
    base_path = os.path.split(LOG_FILE_PATH)[0]
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
        print('The directory {path} was created.'.format(path=base_path))

    with open(LOG_FILE_PATH, 'a') as log:
        log.write("List of missing grid districts:")
        pass
######################################################

# database connection/ session
oedb_engine = db.connection(section='oedb')
session = sessionmaker(bind=oedb_engine)()

SCHEMA = "model_draft"

create_ding0_sql_tables(oedb_engine, SCHEMA)
db_tables_change_owner(oedb_engine, SCHEMA)
# drop_ding0_db_tables(oedb_engine)

# pickle file locations path to RLI_Daten_Flex01 mount
pkl_filepath = "/home/local/RL-INSTITUT/jonas.huber/rli/Daten_flexibel_01/Ding0/20180823154014"


# choose MV Grid Districts to import
grids = list(range(197, 3609))

# generate all the grids and push them to oedb
for grid_no in grids:

    try:
        nw = load_nd_from_pickle(os.path.join(pkl_filepath, 'ding0_grids__{}.pkl'.format(grid_no)))
    except:
        print('Create log entry in: {}'.format(LOG_FILE_PATH))
        with open(LOG_FILE_PATH, 'a') as log:
            log.write('ding0_grids__{}.pkl not present to the current directory\n'.format(grid_no))
            pass

        continue

    # Extract data from network and put it to DataFrames for csv and for oedb
    network = export_network(nw, run_id=20180823154014)

    # Send data to OEDB
    srid = int(nw.config['geo']['srid'])

    # provide run id for pickle upload

    export_all_pkl_to_db(oedb_engine, SCHEMA, network, srid, grid_no)

# db_tables_change_owner(oedb_engine, SCHEMA)