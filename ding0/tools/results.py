"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import pickle
import os
import pandas as pd

from ding0.tools import config as cfg_ding0
from matplotlib import pyplot as plt
import seaborn as sns


def lv_grid_generators_bus_bar(nd):
    """
    Calculate statistics about generators at bus bar in LV grids

    Parameters
    ----------
    nd : ding0.NetworkDing0
        Network container object

    Returns
    -------
    lv_stats : dict
        Dict with keys of LV grid repr() on first level. Each of the grids has
        a set of statistical information about its topology
    """

    lv_stats = {}

    for la in nd._mv_grid_districts[0].lv_load_areas():
        for lvgd in la.lv_grid_districts():

            station_neighbors = list(lvgd.lv_grid._graph[
                                         lvgd.lv_grid._station].keys())

            # check if nodes of a statio are members of list generators
            station_generators = [x for x in station_neighbors
                                  if x in lvgd.lv_grid.generators()]

            lv_stats[repr(lvgd.lv_grid._station)] = station_generators


    return lv_stats


def calculate_mvgd_stats(nodes_df, edges_df):
    """
    Statistics for each MV grid district

    Parameters
    ----------
    nodes_df : pandas.DataFrame
        Statistics on nodes of a MVGD
    edges_df : pandas.DataFrame
        Statistics on edges of a MVGD

    Returns
    -------
    mvgd_stats : pandas.DataFrame
        Dataframe containing several statistical numbers about MVGD

    Notes
    -----
    Power data (i.e. peak load/ generation capacity) is returned in MW
    """

    generators = ['wind', 'solar', 'biomass', 'run_of_river', 'gas',
                  'geothermal']

    mv_generation = nodes_df[nodes_df['type'].isin(generators)].groupby(['grid_id', 'type'])[
        'generation_capacity'].sum()

    # get peak load/generation capacity in kW
    mvgd_stats = nodes_df.groupby('grid_id').sum()[
                     ['peak_load', 'generation_capacity']]

    # add generation capacity per generator type in MV grid
    mv_generation = nodes_df[nodes_df['type'].isin(generators)].groupby(
        ['grid_id', 'type'])['generation_capacity'].sum().to_frame().unstack(level=-1)
    mv_generation.columns = [_[1] if isinstance(_, tuple) else _
                             for _ in mv_generation.columns]
    mvgd_stats = pd.concat([mvgd_stats, mv_generation], axis=1)

    # Cumulative generation capacity in MV
    mvgd_stats['MV generation capacity'] = mvgd_stats[
        list(mvgd_stats.columns[mvgd_stats.columns.isin(generators)])].sum(
        axis=1)

    # Cumulative generation capacity of subjacent LV grid
    stations = nodes_df[nodes_df['type'].isin(['LV Station', 'LV station (aggregated)'])]
    mvgd_stats['LV generation capacity'] = stations['generation_capacity'].sum()

    # Cumulative peak load of subjacent LV grid
    mvgd_stats['LV peak load'] = stations['peak_load'].sum()

    # Nominal voltage of MV grid district
    mvgd_stats['v_nom'] = nodes_df.groupby('grid_id').mean()['v_nom']

    # Cable and overhead lines lengths
    cable_line_km = edges_df['length'].groupby(
        [edges_df['grid_id'], edges_df['type_kind']]).sum().unstack(
        level=-1).fillna(0)
    cable_line_km.columns.name = None
    mvgd_stats[['km_cable', 'km_line']] = cable_line_km

    # Amount of rings
    mvgd_stats['rings'] = nodes_df.groupby('grid_id').mean()['rings']

    # Number of aggr. LA, stations, generators, etc. connected at MV level
    type = nodes_df.groupby(['grid_id', 'type']).count()['node_id'].unstack(
        level=-1).fillna(0)
    type.columns.name = None
    type.columns = [_ + ' count' for _ in type.columns]
    mvgd_stats = pd.concat([mvgd_stats, type], axis=1)

    return mvgd_stats


def save_nd_to_pickle(nd, path='', filename=None):
    """
    Use pickle to save the whole nd-object to disc

    Parameters
    ----------
    nd : NetworkDing0
        Ding0 grid container object
    path : str
        Absolute or relative path where pickle should be saved. Default is ''
        which means pickle is save to PWD
    """

    abs_path = os.path.abspath(path)


def save_nd_to_pickle(nd, path='', filename=None):
    """
    Use pickle to save the whole nd-object to disc
    Parameters
    ----------
    nd : NetworkDing0
        Ding0 grid container object
    path : str
        Absolute or relative path where pickle should be saved. Default is ''
        which means pickle is save to PWD
    """

    abs_path = os.path.abspath(path)

    if len(nd._mv_grid_districts) > 1:
        name_extension = '_{number}-{number2}'.format(
            number=nd._mv_grid_districts[0].id_db,
            number2=nd._mv_grid_districts[-1].id_db)
    else:
        name_extension = '_{number}'.format(number=nd._mv_grid_districts[0].id_db)

    if filename is None:
        filename = "ding0_grids_{ext}.pkl".format(
            ext=name_extension)

    # delete attributes of `nd` in order to make pickling work
    # del nd._config
    del nd._orm

    pickle.dump(nd, open(os.path.join(abs_path, filename),"wb"))


def load_nd_from_pickle(filename=None, path=''):
    """
    Use pickle to save the whole nd-object to disc

    Parameters
    ----------
    filename : str
        Filename of nd pickle
    path : str
        Absolute or relative path where pickle should be saved. Default is ''
        which means pickle is save to PWD

    Returns
    -------
    nd : NetworkDing0
        Ding0 grid container object
    """

    abs_path = os.path.abspath(path)

    if filename is None:
        raise NotImplementedError

    return pickle.load(open(os.path.join(abs_path, filename),"rb"))


def plot_cable_length(stats, plotpath):
    """
    Cable length per MV grid district
    """

    # cable and line kilometer distribution
    f, axarr = plt.subplots(2, sharex=True)
    stats.hist(column=['km_cable'], bins=5, alpha=0.5, ax=axarr[0])
    stats.hist(column=['km_line'], bins=5, alpha=0.5, ax=axarr[1])

    plt.savefig(os.path.join(plotpath,
                             'Histogram_cable_line_length.pdf'))

def plot_generation_over_load(stats, plotpath):
    """

    :param stats:
    :param plotpath:
    :return:
    """

    # Generation capacity vs. peak load
    sns.set_context("paper", font_scale=1.1)
    sns.set_style("ticks")

    # reformat to MW
    stats[['generation_capacity', 'peak_load']] = stats[['generation_capacity',
                                                         'peak_load']] / 1e3

    sns.lmplot('generation_capacity', 'peak_load',
               data=stats,
               fit_reg=False,
               hue='v_nom',
               # hue='Voltage level',
               scatter_kws={"marker": "D",
                            "s": 100},
               aspect=2)
    plt.title('Peak load vs. generation capcity')
    plt.xlabel('Generation capacity in MW')
    plt.ylabel('Peak load in MW')

    plt.savefig(os.path.join(plotpath,
                             'Scatter_generation_load.pdf'))


def plot_km_cable_vs_line(stats, plotpath):
    """

    :param stats:
    :param plotpath:
    :return:
    """

    # Cable vs. line kilometer scatter
    sns.lmplot('km_cable', 'km_line',
               data=stats,
               fit_reg=False,
               hue='v_nom',
               # hue='Voltage level',
               scatter_kws={"marker": "D",
                            "s": 100},
               aspect=2)
    plt.title('Kilometer of cable/line')
    plt.xlabel('Km of cables')
    plt.ylabel('Km of overhead lines')

    plt.savefig(os.path.join(plotpath,
                             'Scatter_cables_lines.pdf'))


def concat_nd_pickles(self, mv_grid_districts):
    """
    Read multiple pickles, join nd objects and save to file

    Parameters
    ----------
    mv_grid_districts : list
        Ints describing MV grid districts
    """

    pickle_name = cfg_ding0.get('output', 'nd_pickle')
    # self.nd = self.read_pickles_from_files(pickle_name)


    # TODO: instead of passing a list of mvgd's, pass list of filenames plus optionally a basth_path
    for mvgd in mv_grid_districts[1:]:

        filename = os.path.join(
            self.base_path,
            'results', pickle_name.format(mvgd))
        if os.path.isfile(filename):
            mvgd_pickle = pickle.load(open(filename, 'rb'))
            if mvgd_pickle._mv_grid_districts:
                mvgd_1.add_mv_grid_district(mvgd_pickle._mv_grid_districts[0])

    # save to concatenated pickle
    pickle.dump(mvgd_1,
                open(os.path.join(
                    self.base_path,
                    'results',
                    "ding0_grids_{0}-{1}.pkl".format(
                        mv_grid_districts[0],
                        mv_grid_districts[-1])),
                    "wb"))

    # save stats (edges and nodes data) to csv
    nodes, edges = mvgd_1.to_dataframe()
    nodes.to_csv(os.path.join(
        self.base_path,
        'results', 'mvgd_nodes_stats_{0}-{1}.csv'.format(
            mv_grid_districts[0], mv_grid_districts[-1])),
        index=False)
    edges.to_csv(os.path.join(
        self.base_path,
        'results', 'mvgd_edges_stats_{0}-{1}.csv'.format(
            mv_grid_districts[0], mv_grid_districts[-1])),
        index=False)


# TODO: old code, that may is used for re-implementation, @gplssm
# that old code was part of the ResultsDing0 class that was removed later
#
# def concat_nd_pickles(self, mv_grid_districts):
#     """
#     Read multiple pickles, join nd objects and save to file
#
#     Parameters
#     ----------
#     mv_grid_districts : list
#         Ints describing MV grid districts
#     """
#
#     pickle_name = cfg_ding0.get('output', 'nd_pickle')
#     # self.nd = self.read_pickles_from_files(pickle_name)
#
#     mvgd_1 = pickle.load(
#         open(os.path.join(
#             self.base_path,
#             'results',
#             pickle_name.format(mv_grid_districts[0])),
#             'rb'))
#     # TODO: instead of passing a list of mvgd's, pass list of filenames plus optionally a basth_path
#     for mvgd in mv_grid_districts[1:]:
#
#         filename = os.path.join(
#             self.base_path,
#             'results', pickle_name.format(mvgd))
#         if os.path.isfile(filename):
#             mvgd_pickle = pickle.load(open(filename, 'rb'))
#             if mvgd_pickle._mv_grid_districts:
#                 mvgd_1.add_mv_grid_district(mvgd_pickle._mv_grid_districts[0])
#
#     # save to concatenated pickle
#     pickle.dump(mvgd_1,
#                 open(os.path.join(
#                     self.base_path,
#                     'results',
#                     "ding0_grids_{0}-{1}.pkl".format(
#                         mv_grid_districts[0],
#                         mv_grid_districts[-1])),
#                     "wb"))
#
#     # save stats (edges and nodes data) to csv
#     nodes, edges = mvgd_1.to_dataframe()
#     nodes.to_csv(os.path.join(
#         self.base_path,
#         'results', 'mvgd_nodes_stats_{0}-{1}.csv'.format(
#             mv_grid_districts[0], mv_grid_districts[-1])),
#         index=False)
#     edges.to_csv(os.path.join(
#         self.base_path,
#         'results', 'mvgd_edges_stats_{0}-{1}.csv'.format(
#             mv_grid_districts[0], mv_grid_districts[-1])),
#         index=False)
#
#
# def concat_csv_stats_files(self, ranges):
#     """
#     Concatenate multiple csv files containing statistics on nodes and edges.
#
#
#     Parameters
#     ----------
#     ranges : list
#         The list contains tuples of 2 elements describing start and end of
#         each range.
#     """
#
#     for f in ['nodes', 'edges']:
#         file_base_name = 'mvgd_' + f + '_stats_{0}-{1}.csv'
#
#         filenames = []
#         [filenames.append(file_base_name.format(mvgd_ids[0], mvgd_ids[1]))
#          for mvgd_ids in ranges]
#
#         results_file = 'mvgd_{0}_stats_{1}-{2}.csv'.format(
#             f, ranges[0][0], ranges[-1][-1])
#
#         self.concat_and_save_csv(filenames, results_file)
#
#
# def concat_and_save_csv(self, filenames, result_filename):
#     """
#     Concatenate and save multiple csv files in `base_path` specified by
#     filnames
#
#     The path specification of files in done via `self.base_path` in the
#     `__init__` method of this class.
#
#
#     Parameters
#     filenames : list
#         Files to be concatenates
#     result_filename : str
#         File name of resulting file
#
#     """
#
#     list_ = []
#
#     for filename in filenames:
#         df = pd.read_csv(os.path.join(self.base_path, 'results', filename),
#                          index_col=None, header=0)
#         list_.append(df)
#
#     frame = pd.concat(list_)
#     frame.to_csv(os.path.join(
#         self.base_path,
#         'results', result_filename), index=False)
#
#
# def read_csv_results(self, concat_csv_file_range):
#     """
#     Read csv files (nodes and edges) containing results figures
#     Parameters
#     ----------
#     concat_csv_file_range : list
#         Ints describe first and last mv grid id
#     """
#
#     self.nodes = pd.read_csv(
#         os.path.join(self.base_path,
#                      'results',
#                      'mvgd_nodes_stats_{0}-{1}.csv'.format(
#                          concat_csv_file_range[0], concat_csv_file_range[-1]
#                      ))
#     )
#
#     self.edges = pd.read_csv(
#         os.path.join(self.base_path,
#                      'results',
#                      'mvgd_edges_stats_{0}-{1}.csv'.format(
#                          concat_csv_file_range[0], concat_csv_file_range[-1]
#                      ))
#     )
