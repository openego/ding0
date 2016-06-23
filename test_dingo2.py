#!/usr/bin/env python3

import matplotlib.pyplot as plt
from oemof import db

from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo

plt.close('all')

cfg_dingo.load_config('config_db_tables')
cfg_dingo.load_config('config_calc')
cfg_dingo.load_config('config_files')
cfg_dingo.load_config('config_misc')

# get engine for database connection
#conn = db.connection(db_section='ontohub_wdb', cfg_file='~/.dingo/config') # <-- TODO: include custom config file from given path (+input for oemof)

# instantiate dingo network object
nd = NetworkDingo(name='network')

# get database connection info from config file
conn = db.connection(section='oedb_remote')

#mv_regions=[360, 571, 593, 368, 491, 425, 416, 372, 387, 407, 403, 373, 482] # some MV regions from SPF region
mv_regions=[482]

nd.import_mv_regions(conn, mv_regions)

conn.close()

nd.mv_routing(debug=False, animation=False)

nd.mv_parametrize_grid()

conn = db.connection(section='oedb_remote')
nd.export_mv_grid(conn, mv_regions)
conn.close()


#df = nx.to_pandas_dataframe(nd._mv_regions[0].mv_grid._graph)
# import pprint
# for edge in nd._mv_regions[0].mv_grid._graph.edge.keys():
#     # print(edge, type(edge))
#     pprint.pprint(edge)
#     pprint.pprint(nd._mv_regions[0].mv_grid._graph.edge[edge])

#nd._mv_regions[0].mv_grid.graph_draw()
