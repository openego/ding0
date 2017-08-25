#!/usr/bin/env python3

"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import matplotlib.pyplot as plt
import oemof.db as db
import time
import os
import pandas as pd
import dingo.tools.results
import itertools

from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo, results

from math import floor, ceil
import multiprocessing as mp



########################################################
def parallel_run(districts_list, n_of_processes, n_of_districts):
    '''Organize parallel runs of dingo.

    The function take all districts in a list and divide them into 
    n_of_processes parallel processes. For each process, the assigned districts
    are given to the function process_runs() with the argument n_of_districts

    Parameters
    ----------
    districts_list: list of int
        List with all districts to be run.
    n_of_processes: int
        Number of processes to run in parallel
    n_of_districts: int
        Number of districts to be run in each cluster given as argument to
        process_stats()
        
    See Also
    --------
    dingo_runs

    '''
    #######################################################################
    # Define an output queue
    output_info = mp.Queue()
    #######################################################################
    # Setup a list of processes that we want to run
    max_dist = len(districts_list)
    threat_long = floor(max_dist / n_of_processes)

    threats = [districts_list[x:x + threat_long] for x in range(0, len(districts_list), threat_long)]

    processes = []
    for th in threats:
        mv_districts = th
        processes.append(mp.Process(target=process_runs,
                                    args=(mv_districts, n_of_districts,
                                          output_info)))
    #######################################################################
    # Run processes
    for p in processes:
        p.start()
    # Exit the completed processes
    for p in processes:
        p.join()
    # Resque output_info from processes
    output = [output_info.get() for p in processes]
    output = list(itertools.chain.from_iterable(output))

    return output

########################################################
def process_runs(mv_districts, n_of_districts, output_info):
    '''Runs a process organized by parallel_run()
    
    The function take all districts mv_districts and divide them into clusters
    of n_of_districts each. For each cluster, dingo is run and the resulting
    network is saved as a pickle

    Parameters
    ----------
    mv_districts: list of int
        List with all districts to be run.
    n_of_districts: int
        Number of districts in a cluster
    output_info:
        Info about how the run went?
    
    See Also
    --------
    parallel_run

    '''
    #######################################################################
    # database connection
    conn = db.connection(section='oedb')
    #############################
    clusters = [mv_districts[x:x + n_of_districts] for x in range(0, len(mv_districts), n_of_districts)]
    output_clusters= []

    for cl in clusters:
        print('\n########################################')
        print('  Running dingo for district', cl)
        print('########################################')
        nw_name = 'network_MVdist_'+str(cl[0])+'_to_'+str(cl[-1])
        nw = NetworkDingo(name=nw_name)
        try:
            msg = nw.run_dingo(conn=conn, mv_grid_districts_no=cl)
            if msg:
                status = 'run error'
            else:
                msg = 'OK'
                status = 'OK'
                results.save_nd_to_pickle(nw, filename=nw_name+'.pkl')
            output_clusters.append((nw_name,status,msg))
        except Exception as e:
            output_clusters.append((nw_name,  'corrupt dist', e))
            continue

    output_info.put(output_clusters)

    #######################################################################
    #close connection and bye bye
    conn.close()

#######################################################################################
if __name__ == '__main__':
    #############################################
    # run in parallel
    mv_grid_districts = list(range(1728, 1740))
    n_of_processes = mp.cpu_count() #number of parallel threaths
    n_of_districts = 3 #n° of districts in each cluster

    out = parallel_run(mv_grid_districts,n_of_processes,n_of_districts)

    print(out)
