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
`example_single_grid_district.py` and use the option to save the `nd` object to
disk. If the example script was executed in PWD, do not change `base_path`
below.
"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from dingo.tools import results
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
    stats = results.calculate_mvgd_stats(nodes_df, edges_df)

    # plot distribution of load/generation of subjacent LV grids
    stations = nodes_df[nodes_df['type'] == 'LV Station']
    f, axarr = plt.subplots(2, sharex=True)
    f.suptitle("Peak load (top)/ peak generation capacity (bottom) at LV "
              "substation in kW")
    stations.hist(column=['peak_load'], bins=20, alpha=0.5, ax=axarr[0])
    axarr[0].set_title("Peak load in kW")
    stations.hist(column=['generation_capacity'], bins=20, alpha=0.5, ax=axarr[1])
    axarr[1].set_title("Peak generation capacity in kW")
    plt.show()

    # Introduction of report
    print("You are analyzing MV grid district {mvgd}\n".format(
        mvgd=int(stats.index.values)))

    # Print peak load
    print("Total peak load: {load:.0f} kW".format(load=float(stats['peak_load'])))
    print("\t thereof MV: {load:.0f} kW".format(load=0))
    print("\t thereof LV: {load:.0f} kW".format(load=float(stats['LV peak load'])))

    # Print generation capacity
    print("\nTotal generation capacity: {gen:.0f} kW".format(
        gen=float(stats['generation_capacity'])))
    print("\t thereof MV: {gen:.0f} kW".format(
        gen=float(stats['MV generation capacity'])))
    print("\t thereof LV: {gen:.0f} kW".format(
        gen=float(stats['LV generation capacity'])))

    # print total length of cables/overhead lines
    print("\nTotal cable length: {length:.1f} km".format(length=float(stats['km_cable'])))
    print("Total line length: {length:.1f} km".format(length=float(stats['km_line'])))

    # Other predefined functions bring extra information for example the number
    # of generators directly connected to the  bus bar of LV station
    stations_generators = results.lv_grid_generators_bus_bar(nd)
    print('\nGenerators directly connected to the substation')
    for k, v in stations_generators.items():
        if v:
            print("{station}: {gens}".format(station=k, gens=len(v)))

    # Number of line/cable equipment type
    print("\n")
    for t, cnt in dict(edges_df.groupby('type_name').size()).items():
        print("Line/cable of type {type} occurs {cnt} times".format(type=t,
                                                                    cnt=cnt))

    # Access results directly from nd-object if they are not available in stats/
    # nodes_df or edges_df
    # One example: print length of each half ring in this MVGD
    print("\n")
    root = nd._mv_grid_districts[0].mv_grid.station()
    for circ_breaker in nd._mv_grid_districts[0].mv_grid.circuit_breakers():
        for half_ring in [0, 1]:
            half_ring_length = nd._mv_grid_districts[
                                   0].mv_grid.graph_path_length(
                circ_breaker.branch_nodes[half_ring], root) / 1000
            print('Length to circuit breaker', repr(circ_breaker),
                  ', half-ring', str(half_ring), ':', str(half_ring_length),
                  'km')

if __name__ == '__main__':
    filename = 'dingo_grids_example.pkl'
    example_stats(filename)