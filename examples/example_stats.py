#!/usr/bin/env python3

"""This is a simple example file for DINGO.

__copyright__ = "Reiner Lemoine Institut, openego development group"
__license__ = "GNU GPLv3"
__author__ = "Jonathan Amme, Guido Pleßmann"
"""

import matplotlib.pyplot as plt
import oemof.db as db
import time
# import objgraph

from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo

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

#mv_grid_districts = [386,372,406,371,402,415,480,424,489,367,359,569,591]
#mv_grid_districts = [402, 406, 489, 480, 371]
#mv_grid_districts=[1328]
#mv_grid_districts=[1294]
#mv_grid_districts=[419]
#mv_grid_districts = [359, 415, 424, 447, 402, 406, 489, 480, 371]
#mv_grid_districts=[359]
#mv_grid_districts = [386,372,406,371,402,415,480,424,489,367]#,569,359,591]
#mv_grid_districts=[3087, 2990, 3080, 3034, 3088]
#mv_grid_districts=[3080]#, 3080]#, 3080]
mvgd_exclude = [56]
mvgd_first = 578
mvgd_last = 580

mv_grid_districts = [mv for mv in list(range(mvgd_first, mvgd_last)) if
                     mv not in mvgd_exclude]
# mv_grid_districts = list(range(1,100))

nd.import_pf_config()

nd.import_mv_grid_districts(conn, mv_grid_districts)

nd.import_generators(conn)

nd.mv_parametrize_grid()

nd.validate_grid_districts()

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

# nd.export_mv_grid(conn, mv_grid_districts)
nd.export_mv_grid_new(conn, mv_grid_districts)

import pickle
pickle.dump(nd,
            open("dingo_grids_{0}-{1}.pkl".format(mvgd_first, mvgd_last), "wb"))
mvgd_stats = nd.to_dataframe(conn, mv_grid_districts)
print(mvgd_stats)
# mvgd_stats.to_hdf('mvgd_stats_{0}-{1}.hf5'.format(mvgd_first, mvgd_last), 'data')
mvgd_stats.to_csv('mvgd_stats_{0}-{1}.csv'.format(mvgd_first, mvgd_last))

conn.close()
#objgraph.show_refs([nd], filename='nd.png')
print('Elapsed time for', str(len(mv_grid_districts)), 'MV grid districts (seconds): {}'.format(time.time() - start))

# reinforce MV grid
#nd.reinforce_grid()
