#!/usr/bin/env python3

import matplotlib.pyplot as plt
from oemof import db

from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo

plt.close('all')

cfg_dingo.load_config('config_db_tables')
cfg_dingo.load_config('config_calc')

# get engine for database connection
#conn = db.connection(db_section='ontohub_wdb', cfg_file='~/.dingo/config') # <-- TODO: include custom config file from given path (+input for oemof)
#conn = db.connection(db_section='ontohub_wdb')

# instantiate dingo network object
nd = NetworkDingo(name='network')

# get database connection info from config file
conn = db.connection(section='oedb')

#mv_regions=[106, 125, 500, 722, 887, 1049] # some MV regions from SPF region
mv_regions=[106] # some MV regions from SPF region

nd.import_mv_regions(conn, mv_regions)


# cre
# create_lv_stations(network)

conn.close()

nd.mv_routing()

#df = nx.to_pandas_dataframe(nd._mv_regions[0].mv_grid._graph)

nd._mv_regions[0].mv_grid.graph_draw()