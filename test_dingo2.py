#!/usr/bin/env python3

import matplotlib.pyplot as plt
from oemof import db
conn = db.connection(section='oedb')
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

# mv_grid_districts=[360, 571, 593, 368, 491, 425, 416, 372, 387, 407, 403, 373, 482] # some MV grid_districts from SPF region
# mv_grid_districts=[360, 571, 593, 368, 491, 416, 372, 387, 407, 403, 373, 482] # some MV grid_districts from SPF region
# mv_grid_districts=[482]
# mv_grid_districts = [386,372,406,371,402,415,480,424,489,367,359,569,591]
mv_grid_districts=[489]

nd.import_mv_grid_districts(conn, mv_grid_districts)
nd.import_generators(conn)

nd.mv_parametrize_grid()

nd.mv_routing(debug=False, animation=False)

nd.connect_generators()

nd.set_branch_ids()

# Open and close all circuit breakers in grid (for testing)
[gd.mv_grid.open_circuit_breakers() for gd in nd._mv_grid_districts]
#nd._mv_grid_districts[0].mv_grid.close_circuit_breakers()

for mv_grid_district in nd._mv_grid_districts:
    mv_grid_district.mv_grid.export_to_pypsa(conn)
    mv_grid_district.mv_grid.run_powerflow(conn)
    mv_grid_district.mv_grid.import_powerflow_results(conn)

nd.export_mv_grid(conn, mv_grid_districts)

conn.close()

# for edge in nd._mv_grid_districts[0].mv_grid.graph_edges():
#     if edge['branch'].type is not None:
#         print(edge['branch'].type['name'])
#     else:
#         print('None')

# lvrg = []
# for mv_grid_district in nd.mv_grid_districts():
#     #print(mv_grid_district._lv_load_area_groups)
#     #print(type(mv_grid_district._lv_load_area_groups))
#     for lv_load_area_group in iter(mv_grid_district._lv_load_area_groups):
#         lvrg.append([str(lv_load_area_group), lv_load_area_group.peak_load_sum, lv_load_area_group.branch_length_sum])
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
