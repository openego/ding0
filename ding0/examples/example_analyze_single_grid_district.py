#!/usr/bin/env python3

"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io

Notes
-----

This example file assumes you have already run the example file
`example_single_grid_district.py` and use the option to save the `nd` object to
disk. If the example script was executed in PWD, do not change `base_path`
below.
"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "nesnoj, gplssm"

from ding0.tools import results
from pandas import option_context
from matplotlib import pyplot as plt

base_path = ''


def example_stats(filename, plotpath=''):
    """
    Obtain statistics from create grid topology

    Prints some statistical numbers and produces exemplary figures
    """

    nd = results.load_nd_from_pickle(filename=filename)

    nodes_df, edges_df = nd.to_dataframe()

    # get statistical numbers about grid
    stats = results.calculate_mvgd_stats(nd)

    # plot distribution of load/generation of subjacent LV grids
    stations = nodes_df[nodes_df['type'] == 'LV Station']
    f, axarr = plt.subplots(2, sharex=True)
    f.suptitle("Peak load (top)/ peak generation capacity (bottom) at LV "
               "substation in kW")
    stations['peak_load'].hist(bins=20, alpha=0.5, ax=axarr[0])
    axarr[0].set_title("Peak load in kW")
    stations['generation_capacity'].hist(bins=20, alpha=0.5, ax=axarr[1])
    axarr[1].set_title("Peak generation capacity in kW")
    plt.show()

    # Introduction of report
    print("You are analyzing MV grid district {mvgd}\n".format(
        mvgd=int(stats.index.values)))

    # print all the calculated stats
    # this isn't a particularly beautiful format but it is
    # information rich
    with option_context('display.max_rows', None, 'display.max_columns', None):
        print(stats.T)


if __name__ == '__main__':
    filename = 'ding0_grids_example.pkl'
    example_stats(filename)
