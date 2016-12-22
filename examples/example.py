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

# database connection
conn = db.connection(section='oedb')

# instantiate dingo network object
nd = NetworkDingo(name='network')

# mv_grid_districts = [386,372,406,371,402,415,480,424,489,367,359,569,591]
mv_grid_districts=[480]

nd.import_mv_grid_districts(conn, mv_grid_districts)

nd.import_generators(conn)

nd.mv_parametrize_grid()

nd.mv_routing(debug=False, animation=False)

nd.connect_generators()

nd.set_branch_ids()

# DEBUG (Compare graphs to CLEAN UP THE SALT)
#conn.close()
#from dingo.tools.debug import compare_graphs
#compare_graphs(graph=nd._mv_grid_districts[0].mv_grid._graph,
#               mode='compare')

nd.set_circuit_breakers()

# Open and close all circuit breakers in grid
[gd.mv_grid.open_circuit_breakers() for gd in nd._mv_grid_districts]

# Analyze grid by power flow analysis
for mv_grid_district in nd._mv_grid_districts:
    mv_grid_district.mv_grid.run_powerflow(conn, method='onthefly')


nd.export_mv_grid(conn, mv_grid_districts)

conn.close()

# reinforce MV grid
#nd.reinforce_grid()
