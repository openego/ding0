#!/usr/bin/env python3

"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import time
from datetime import datetime
import os
import itertools

from ding0.core import NetworkDing0
from ding0.tools import results, db

from math import floor
import multiprocessing as mp
import pandas as pd
import json


BASEPATH = os.path.join(os.path.expanduser('~'), '.ding0')


########################################################
def parallel_run(districts_list, n_of_processes, n_of_districts, run_id,
                 base_path=None):
    '''Organize parallel runs of ding0.

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
    run_id: str
        Identifier for a run of Ding0. For example it is used to create a
        subdirectory of os.path.join(`base_path`, 'results')
    base_path : str
        Base path for ding0 data (input, results and logs).
        Default is `None` which sets it to :code:`~/.ding0` (may deviate on
        windows systems).
        Specify your own but keep in mind that it a required a particular
        structure of subdirectories.
        
    See Also
    --------
    ding0_runs

    '''

    # define base path
    if base_path is None:
        base_path = BASEPATH

    if not os.path.exists(os.path.join(base_path, run_id)):
        os.makedirs(os.path.join(base_path, run_id))

    start = time.time()
    #######################################################################
    # Define an output queue
    output_info = mp.Queue()
    #######################################################################
    # Setup a list of processes that we want to run
    max_dist = len(districts_list)
    threat_long = floor(max_dist / n_of_processes)

    if threat_long == 0:
        threat_long = 1

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
    # Resque output_info from processes
    output = [output_info.get() for p in processes]
    output = list(itertools.chain.from_iterable(output))
    # Exit the completed processes
    for p in processes:
        p.join()

    #######################################################################
    print('Elapsed time for', str(max_dist),
          'MV grid districts (seconds): {}'.format(time.time() - start))

    return output

########################################################
def process_runs(mv_districts, n_of_districts, output_info):
    '''Runs a process organized by parallel_run()
    
    The function take all districts mv_districts and divide them into clusters
    of n_of_districts each. For each cluster, ding0 is run and the resulting
    network is saved as a pickle

    Parameters
    ----------
    mv_districts: list of int
        List with all districts to be run.
    n_of_districts: int
        Number of districts in a cluster
    output_info:
        Info about how the run went
    run_id: str
        Identifier for a run of Ding0. For example it is used to create a
        subdirectory of os.path.join(`base_path`, 'results')
    base_path : str
        Base path for ding0 data (input, results and logs).
        Default is `None` which sets it to :code:`~/.ding0` (may deviate on
        windows systems).
        Specify your own but keep in mind that it a required a particular
        structure of subdirectories.
    
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
        print('  Running ding0 for district', cl)
        print('########################################')

        nw_name = 'ding0_grids_' + str(cl[0])
        if not cl[0] == cl[-1]:
            nw_name = nw_name+'_to_'+str(cl[-1])
        nw = NetworkDing0(name=nw_name)
        try:
            msg = nw.run_ding0(conn=conn, mv_grid_districts_no=cl)
            if msg:
                status = 'run error'
            else:
                msg = ''
                status = 'OK'
                results.save_nd_to_pickle(nw, os.path.join(base_path, run_id))
            output_clusters.append((nw_name,status,msg, nw.metadata))
        except Exception as e:
            output_clusters.append((nw_name,  'corrupt dist', e, nw.metadata))
            continue

    output_info.put(output_clusters)


    #######################################################################
    #close connection and bye bye
    conn.close()

def process_metadata(meta):
    """
    Merge metadata of run on multiple grid districts

    Parameters
    ----------
    meta: list of dict
        Metadata of run of each MV grid district

    Returns
    -------
    dict
        Single metadata dict including merge metadata
    """
    mvgds = []

    metadata = meta[0]

    for mvgd in meta:
        if isinstance(mvgd['mv_grid_districts'], list):
            mvgds.extend(mvgd['mv_grid_districts'])
        else:
            mvgds.append(mvgd['mv_grid_districts'])

    metadata['mv_grid_districts'] = mvgds

    return metadata



if __name__ == '__main__':
    # define individual base path
    base_path = ''#'/home/guido/mnt/rli-daten/Ding0/'

    # set run_id to current timestamp
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")

    # run in parallel
    mv_grid_districts = list(range(1, 3609))
    n_of_processes = mp.cpu_count() #number of parallel threaths
    n_of_districts = 1 #n° of districts in each serial cluster

    out = parallel_run(mv_grid_districts, n_of_processes, n_of_districts,
                       run_id, base_path=base_path)

    # report on unsuccessful runs
    corrupt_out = [_[0:3] for _ in out if not _[1]=='OK']

    corrupt_grid_districts = pd.DataFrame(corrupt_out,
                                          columns=['grid', 'status', 'message'])
    corrupt_grid_districts.to_csv(
        os.path.join(
            base_path,
            run_id,
            'corrupt_mv_grid_districts.txt'),
        index=False,
        float_format='%.0f')

    # save metadata
    meta_dict_list = [_[3] for _ in out]
    metadata = process_metadata(meta_dict_list)
    with open(os.path.join(base_path, run_id, 'Ding0_{}.meta'.format(run_id)),
              'w') as f:
        json.dump(metadata, f)
