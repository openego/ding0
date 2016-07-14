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
conn = db.connection(section='oedb')

#mv_grid_districts=[360, 571, 593, 368, 491, 425, 416, 372, 387, 407, 403, 373, 482] # some MV regions from SPF region
mv_grid_districts=[482]

nd.import_mv_grid_districts(conn, mv_grid_districts)

conn.close()

nd.mv_routing(debug=False, animation=False)

nd.mv_parametrize_grid()

conn = db.connection(section='oedb')
nd.export_mv_grid(conn, mv_grid_districts)
conn.close()

# lvrg = []
# for mv_grid_district in nd.mv_grid_districts():
#     #print(mv_grid_district._lv_region_groups)
#     #print(type(mv_grid_district._lv_region_groups))
#     for lv_region_group in iter(mv_grid_district._lv_region_groups):
#         lvrg.append([str(lv_region_group), lv_region_group.peak_load_sum, lv_region_group.branch_length_sum])
# lvrg = sorted(lvrg, key=lambda x: x[1])
#
# for lvrg_name, lvrg_load, lvrg_length in lvrg:
#     print(lvrg_name, lvrg_load, lvrg_length)



#df = nx.to_pandas_dataframe(nd._mv_grid_districts[0].mv_grid._graph)
# import pprint
# for edge in nd._mv_grid_districts[0].mv_grid._graph.edge.keys():
#     # print(edge, type(edge))
#     pprint.pprint(edge)
#     pprint.pprint(nd._mv_grid_districts[0].mv_grid._graph.edge[edge])

#nd._mv_grid_districts[0].mv_grid.graph_draw()
