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
    export_all_dataframes_to_db, db_tables_change_owner, drop_ding0_db_tables
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
grids = list(range(1, 6))

# generate all the grids and push them to oedb
for grid_no in grids:

    nw = load_nd_from_pickle(os.path.join(pkl_filepath,
                            'ding0_grids__{}.pkl'.format(grid_no)))

    # Extract data from network and put it to DataFrames for csv and for oedb
    run_id, nw_metadata, \
    lv_grid, lv_gen, lv_cd, lv_stations, mvlv_trafos, lv_loads, \
    mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, hvmv_trafos, mv_loads, \
    lines, mvlv_mapping = export_network(nw)

    df_list = [lv_grid, lv_gen, lv_cd, lv_stations, mvlv_trafos, lv_loads,
                mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, hvmv_trafos, mv_loads,
                lines, mvlv_mapping]

    # Send data to OEDB
    srid = str(int(nw.config['geo']['srid']))
    metadata_json = json.loads(nw_metadata)

    export_all_dataframes_to_db(oedb_engine, SCHEMA, nw_metadata, df_list)

# db_tables_change_owner(oedb_engine, SCHEMA)