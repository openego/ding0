#!/usr/bin/env python3

import matplotlib.pyplot as plt
import oemof.db as db
import time
# import objgraph

from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo
import os
import pandas as pd

plt.close('all')
cfg_dingo.load_config('config_db_tables.cfg')
cfg_dingo.load_config('config_calc.cfg')
cfg_dingo.load_config('config_files.cfg')
cfg_dingo.load_config('config_misc.cfg')

start = time.time()

# database connection
conn = db.connection(section='oedb')

# instantiate dingo network object
nd = NetworkDingo(name='network')

base_path = "/home/guido/rli_local/dingo_results/"

mvgd_first = 5
mvgd_last = 6

mvgd_exclude = pd.read_csv(
    os.path.join(
        base_path, 'info', 'corrupt_mv_grid_districts_{0}-{1}.txt'.format(
                              mvgd_first, mvgd_last[-1])
    ))['id'].tolist()
mvgd_exclude = []

mv_grid_districts = [mv for mv in list(range(mvgd_first, mvgd_last)) if
                     mv not in mvgd_exclude]

# mv_grid_districts = list(range(1,100))

nd.import_pf_config()

nd.import_mv_grid_districts(conn, mv_grid_districts)

nd.import_generators(conn)

nd.mv_parametrize_grid()

msg = nd.validate_grid_districts()

nd.mv_routing(debug=False, animation=False)

nd.connect_generators()

nd.set_branch_ids()

# DEBUG (Compare graphs to CLEAN UP THE SALT)
#conn.close()
#from dingo.tools.debug import compare_graphs
#compare_graphs(graph=nd._mv_grid_districts[0].mv_grid._graph,
#               mode='compare')

nd.set_circuit_breakers()

# Open all circuit breakers in grid
nd.control_circuit_breakers(mode='open')

# Analyze grid by power flow analysis
nd.run_powerflow(conn, method='onthefly', export_pypsa=False)

# reinforce MV grid
nd.reinforce_grid()

# nd.export_mv_grid(conn, mv_grid_districts)
nd.export_mv_grid_new(conn, mv_grid_districts)

import pickle
pickle.dump(nd,
            open("dingo_grids_{0}-{1}.pkl".format(mvgd_first, mvgd_last), "wb"))
nodes_stats, edges_stats = nd.to_dataframe()
# mvgd_stats.to_hdf('mvgd_stats_{0}-{1}.hf5'.format(mvgd_first, mvgd_last), 'data')
nodes_stats.to_csv('mvgd_nodes_stats_{0}-{1}.csv'.format(mvgd_first, mvgd_last),
                   index=False)
edges_stats.to_csv('mvgd_edges_stats_{0}-{1}.csv'.format(mvgd_first, mvgd_last),
                   index=False)


conn.close()
#objgraph.show_refs([nd], filename='nd.png')
print('Elapsed time for', str(len(mv_grid_districts)), 'MV grid districts (seconds): {}'.format(time.time() - start))

