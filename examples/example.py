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
from dingo.tools.logger import setup_logger

logger = setup_logger()

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
#mv_grid_districts=[480]
#mv_grid_districts=[632,388,477,474,376,1524,1566,3101,2993,3360,2270]
#mv_grid_districts=[3605,3545,428,3115,206,1199,3242,189]
#mv_grid_districts=[428]
mv_grid_districts=[3545]
#mv_grid_districts = [359, 415, 424, 447, 402, 406, 489, 480, 371]
#mv_grid_districts=[359]
#mv_grid_districts = [386,372,406,371,402,415,480,424,489,367]#,569,359,591]
#mv_grid_districts=[3087, 2990, 3080, 3034, 3088]
#mv_grid_districts=[3080]#, 3080]#, 3080]
#mv_grid_districts = list(range(1250,1351))

nd.import_mv_grid_districts(conn, mv_grid_districts)

nd.import_generators(conn)

nd.mv_parametrize_grid()

nd.validate_grid_districts()

nd.mv_routing(debug=False, animation=False)

nd.connect_generators()

nd.set_branch_ids()

nd.set_circuit_breakers()

# Open all circuit breakers in grid
nd.control_circuit_breakers(mode='open')

# Analyze grid by power flow analysis
nd.run_powerflow(conn, method='onthefly', export_pypsa=False)

# reinforce MV grid
nd.reinforce_grid()

#objgraph.show_refs([nd], filename='nd.png')
logger.info('Elapsed time for {0} MV grid districts (seconds): {1}'.format(
    str(len(mv_grid_districts)), time.time() - start))

# export grids
nd.export_mv_grid(conn, mv_grid_districts)
nd.export_mv_grid_new(conn, mv_grid_districts)

conn.close()

# branch types
branch_types = []
for b in nd._mv_grid_districts[0].mv_grid.graph_edges():
    branch_types.append(b['branch'].type['name'])
print(branch_types)

# MV generation capacity
mv_cum_capacity = 0
for geno in nd._mv_grid_districts[0].mv_grid.generators():
    mv_cum_capacity += geno.capacity
print('Cum. capacity of MV generators:', str(mv_cum_capacity))

# LV generation capacity
from dingo.core.network.stations import LVStationDingo
lv_cum_capacity = 0
for node in nd._mv_grid_districts[0].mv_grid._graph.nodes():
    if isinstance(node, LVStationDingo):
        lv_cum_capacity += node.peak_generation
print('Cum. capacity of LV generators:', str(lv_cum_capacity))

# half ring lengths
root = nd._mv_grid_districts[0].mv_grid.station()
for circ_breaker in nd._mv_grid_districts[0].mv_grid.circuit_breakers():
    for half_ring in [0,1]:
        half_ring_length = nd._mv_grid_districts[0].mv_grid.graph_path_length(circ_breaker.branch_nodes[half_ring], root) / 1000
        print('Length to circuit breaker', repr(circ_breaker), ', half-ring', str(half_ring), ':', str(half_ring_length), 'km')
