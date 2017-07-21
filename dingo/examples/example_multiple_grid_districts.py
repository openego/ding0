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
# import oemof.db as db
import time
import os
import pandas as pd

from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo, results, db

plt.close('all')
cfg_dingo.load_config('config_db_tables.cfg')
cfg_dingo.load_config('config_calc.cfg')
cfg_dingo.load_config('config_files.cfg')
cfg_dingo.load_config('config_misc.cfg')

BASEPATH = os.path.join(os.path.expanduser('~'), '.dingo')


def create_results_dirs(base_path):
    """Create base path dir and subdirectories

    Parameters
    ----------
    base_path : str
        The base path has subdirectories for raw and processed results
    """


    if not os.path.exists(base_path):
        print("Creating directory {} for results data.".format(base_path))
        os.mkdir(base_path)
    if not os.path.exists(os.path.join(base_path, 'results')):
        os.mkdir(os.path.join(base_path, 'results'))
    if not os.path.exists(os.path.join(base_path, 'plots')):
        os.mkdir(os.path.join(base_path, 'plots'))
    if not os.path.exists(os.path.join(base_path, 'info')):
        os.mkdir(os.path.join(base_path, 'info'))
    if not os.path.exists(os.path.join(base_path, 'log')):
        os.mkdir(os.path.join(base_path, 'log'))


def run_multiple_grid_districts(mv_grid_districts, failsafe=False, base_path=None):
    """
    Perform dingo run on given grid districts

    Parameters
    ----------
    mv_grid_districs : list
        Integers describing grid districts
    failsafe : bool
        Setting to True enables failsafe mode where corrupt grid districts
        (mostly due to data issues) are reported and skipped. Report is to be
         found in the log dir under :code:`~/.dingo` . Default is False.
    base_path : str
        Base path for dingo data (input, results and logs).
        Default is `None` which sets it to :code:`~/.dingo` (may deviate on
        windows systems).
        Specify your own but keep in mind that it a required a particular
        structure of subdirectories.

    Returns
    -------
    msg : str
        Traceback of error computing corrupt MV grid district
        .. TODO: this is only true if try-except environment is moved into this
            fundion and traceback return is implemented

    Notes
    -----
    Consider that a large amount of MV grid districts may take hours or up to
    days to compute. A computational run for a single grid district may consume
    around 30 secs.
    """
    start = time.time()

    # define base path
    if base_path is None:
        base_path = BASEPATH

    # database connection
    conn = db.connection(section='oedb')

    corrupt_grid_districts = pd.DataFrame(columns=['id', 'message'])

    for mvgd in mv_grid_districts:
        # instantiate dingo  network object
        nd = NetworkDingo(name='network')

        if not failsafe:
            # run DINGO on selected MV Grid District
            msg = nd.run_dingo(conn=conn,
                         mv_grid_districts_no=[mvgd])

            # save results
            results.save_nd_to_pickle(nd, os.path.join(base_path, 'results'))
        else:
            # try to perform dingo run on grid district
            try:
                msg = nd.run_dingo(conn=conn,
                         mv_grid_districts_no=[mvgd])
                # if not successful, put grid district to report
                if msg:
                    corrupt_grid_districts = corrupt_grid_districts.append(
                        pd.Series({'id': mvgd,
                                   'message': msg[0]}),
                        ignore_index=True)
                # if successful, save results
                else:
                    results.save_nd_to_pickle(nd, os.path.join(base_path,
                                                               'results'))
            except Exception as e:
                corrupt_grid_districts = corrupt_grid_districts.append(
                    pd.Series({'id': mvgd,
                               'message': e}),
                    ignore_index=True)

                continue

        # Merge metadata of multiple runs
        if 'metadata' not in locals():
            metadata = nd.metadata
        else:
            metadata['mv_grid_districts'].extend(mvgd)

    # report on unsuccessful runs
    corrupt_grid_districts.to_csv(
        os.path.join(
            base_path,
            'info',
            'corrupt_mv_grid_districts.txt'),
        index=False,
        float_format='%.0f')

    conn.close()
    print('Elapsed time for', str(len(mv_grid_districts)),
          'MV grid districts (seconds): {}'.format(time.time() - start))


    return msg


if __name__ == '__main__':

    # create directories for local results data
    create_results_dirs(BASEPATH)

    # define grid district by its id (int)
    mv_grid_districts = list(range(1, 3608))

    # run grid districts
    run_multiple_grid_districts(mv_grid_districts, failsafe=True, base_path='/home/guido/git-repos/dingo/tmp_dingo')
