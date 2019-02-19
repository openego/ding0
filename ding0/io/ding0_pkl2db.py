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


# define logger
logger = setup_logger()

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
grids = list(range(43, 3609))

# generate all the grids and push them to oedb
for grid_no in grids:

    nw = load_nd_from_pickle(os.path.join(pkl_filepath, 'ding0_grids__{}.pkl'.format(grid_no)))

    # Extract data from network and put it to DataFrames for csv and for oedb
    network = export_network(nw, run_id=20180823154014)

    # Send data to OEDB
    srid = int(nw.config['geo']['srid'])

    # provide run id for pickle upload

    export_all_pkl_to_db(oedb_engine, SCHEMA, network, srid, grid_no)

# db_tables_change_owner(oedb_engine, SCHEMA)