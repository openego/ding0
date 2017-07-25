#!/usr/bin/env python3

"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io

Notes
-----

This example file assumes you have already run the example file
`example_multiple_grid_districts.py` and use the option to save the `nd` object to
disc. If the example script was executed in PWD, do not change `base_path`
below.
"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from dingo.tools import results
from dingo.tools.logger import get_default_home_dir
import os
import pandas as pd
from matplotlib import pyplot as plt


BASEPATH = get_default_home_dir()

def dingo_exemplary_plots(stats, base_path=BASEPATH):
    """
    Analyze multiple grid district data generated with Dingo

    Parameters
    ----------
    stats : pandas.DataFrame
        Statistics of each MV grid districts
    base_path : str
        Root directory of Dingo data structure, i.e. '~/.dingo' (which is
        default).
    """

    # make some plot
    plotpath = os.path.join(base_path, 'plots')
    results.plot_cable_length(stats, plotpath)
    results.plot_generation_over_load(stats, plotpath)
    results.plot_km_cable_vs_line(stats, plotpath)


def nd_load_and_stats(filenames, base_path=BASEPATH):
    """
    Load multiple files from disk and generate stats

    Passes the list of files assuming the dingo data structure as default in
    :code:`~/.dingo`.
    Data will concatenated and key indicators for each grid district are
    returned in table and graphic format

    Parameters
    ----------
    filenames : list of str
        Provide list of files you want to analyze
    base_path : str
        Root directory of Dingo data structure, i.e. '~/.dingo' (which is
        default).
    Returns
    -------
    stats : pandas.DataFrame
        Statistics of each MV grid districts
    """

    # load Dingo data
    nds = []
    for filename in filenames:
        try:
            nd_load = results.load_nd_from_pickle(filename=
                                             os.path.join(base_path,
                                                          'results',
                                                          filename))

            nds.append(nd_load)
        except:
            print("File {mvgd} not found. It was maybe excluded by Dingo or "
                  "just forgotten to generate by you...".format(mvgd=filename))

    nd = nds[0]

    for n in nds[1:]:
        nd.add_mv_grid_district(n._mv_grid_districts[0])

    nodes_df, edges_df = nd.to_dataframe()

    # get statistical numbers about grid
    stats = results.calculate_mvgd_stats(nodes_df, edges_df)

    # TODO: correct LV peak load/ generation capacity. Same in all LV GD
    return stats

if __name__ == '__main__':
    base_path = BASEPATH

    mv_grid_districts = list(range(1, 20))

    filenames = ["dingo_grids__{ext}.pkl".format(ext=_)
                 for _ in mv_grid_districts]

    # load files and generate statistical number about each mv grid district
    stats = nd_load_and_stats(filenames)

    # save stats file to disc
    stats.to_csv(os.path.join(base_path, 'results',
                              'dingo_grids_stats_{first}-{last}'.format(
                                  first=mv_grid_districts[0],
                                  last=mv_grid_districts[-1])))

    # load stats from file
    # stats = pd.read_csv(os.path.join(base_path, 'results',
    #                             'dingo_grids_stats_{first}-{last}'.format(
    #                                 first=mv_grid_districts[0],
    #                                 last=mv_grid_districts[-1])))

    # make some plots to compare grid districts
    dingo_exemplary_plots(stats)
