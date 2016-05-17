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

# instantiate dingo network object
nd = NetworkDingo(name='network')

# get database connection info from config file
conn = db.connection(db_section='oedb')

#mv_regions=[106, 125, 500, 722, 887, 1049] # some MV regions from SPF region
mv_regions=[217] # some MV regions from SPF region

nd.import_mv_regions(conn, mv_regions)



# cre
# create_lv_stations(network)

conn.close()

nd.mv_routing()

nd.mv_parametrize_grid()

#df = nx.to_pandas_dataframe(nd._mv_regions[0].mv_grid._graph)
# import pprint
# for edge in nd._mv_regions[0].mv_grid._graph.edge.keys():
#     # print(edge, type(edge))
#     pprint.pprint(edge)
#     pprint.pprint(nd._mv_regions[0].mv_grid._graph.edge[edge])
nd._mv_regions[0].mv_grid.graph_draw()
