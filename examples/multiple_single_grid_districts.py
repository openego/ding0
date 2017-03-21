#!/usr/bin/env python3

"""This is a simple example file for DINGO.

__copyright__ = "Reiner Lemoine Institut, openego development group"
__license__ = "GNU GPLv3"
__author__ = "Jonathan Amme, Guido Pleßmann"
"""

import matplotlib.pyplot as plt
import oemof.db as db
import time
import pickle

from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo

plt.close('all')
cfg_dingo.load_config('config_db_tables.cfg')
cfg_dingo.load_config('config_calc.cfg')
cfg_dingo.load_config('config_files.cfg')
cfg_dingo.load_config('config_misc.cfg')


def run_dingo(mv_grid_districs):
    """
    Perform dingo run on given grid districts

    Parameters
    ----------
    mv_grid_districs : list
    Integers describing grid districts
    """

    # instantiate dingo  network object
    nd = NetworkDingo(name='network')

    nd.import_pf_config()

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

    # nd.export_mv_grid(conn, mv_grid_districts)
    nd.export_mv_grid_new(conn, mv_grid_districts)

    pickle.dump(nd,
                open("dingo_grids_{0}-{1}.pkl".format(mvgd_first, mvgd_last), "wb"))


if __name__ == '__main__':

    start = time.time()

    # database connection
    conn = db.connection(section='oedb')

    mvgd_exclude = []
    mvgd_first = 1
    mvgd_last = 3608

    mv_grid_districts = [mv for mv in list(range(mvgd_first, mvgd_last)) if
                             mv not in mvgd_exclude]

    corrupt_grid_districts = []

    for mv_grid_district in mv_grid_districts:
        try:
            run_dingo(mv_grid_district)
        except:
            corrupt_grid_districts.append(mv_grid_district)

    print(corrupt_grid_districts)

    thefile = open('corrupt_mv_grid_districts.txt', 'w')
    for item in corrupt_grid_districts:
        thefile.write("%s\n" % item)


    conn.close()
    print('Elapsed time for', str(len(mv_grid_districts)),
          'MV grid districts (seconds): {}'.format(time.time() - start))

    # reinforce MV grid
    #nd.reinforce_grid()
