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


import pickle
import os
import pandas as pd

from dingo.tools import config as cfg_dingo
from matplotlib import pyplot as plt
import seaborn as sns


class ResultsDingo:
    # TODO: check DOCSTRING in class and methods
    """Holds raw results data and provides methods to generate a set of results

    `base_path` is optional, but it's actually required
    
    Parameters
    ----------
    base_path : str
        Path to results dir structure
            
    """

    def __init__(self, base_path):
        """
        """

        # if mv_grid_districts is not None:
        #     self.mv_grid_districts_including_invalid = mv_grid_districts
        # else:
        #     self.mv_grid_districts_including_invalid = None


        self.base_path = base_path

        self.edges = None
        self.nodes = None
        self.global_stats = None
        self.mvgd_stats = None

        # if os.path.isfile(os.path.join(self.base_path,
        #                  'info',
        #                   'corrupt_mv_grid_districts_{0}-{1}.txt'.format(
        #                       mv_grid_districts[0], mv_grid_districts[-1]))):
        #     self.invalid_mvgd = pd.read_csv(
        #         os.path.join(self.base_path,
        #                      'info',
        #                      'corrupt_mv_grid_districts_{0}-{1}.txt'.format(
        #                       mv_grid_districts[0], mv_grid_districts[-1])))

        # # get list of excluded grids (invalid) from info file
        # self.excluded_grid_districts = self.invalid_mvgd['id'].tolist()

        # # define currently valid mv_grid districts
        # if self.mv_grid_districts_including_invalid is not None:
        #     self.mv_grid_districs = [
        #         mv for mv in self.mv_grid_districts_including_invalid
        #         if mv not in self.excluded_grid_districts]

        # read results data from a single file
        # if filenames is not None:
        #     if 'nd' in list(filenames.keys()):
        #         self.nd = self.read_nd_multiple_mvgds(filenames['nd'])
        #     if 'edges' in list(filenames.keys()):
        #         self.edges = pd.read_csv(os.path.join(self.base_path,
        #                                               'results',
        #                                               filenames['edges']))
        #     if 'nodes' in list(filenames.keys()):
        #         self.nodes = pd.read_csv(os.path.join(self.base_path,
        #                                               'results',
        #                                               filenames['nodes']))
        # read results from single file per mv grid district
        # elif mv_grid_districts is not None:
        #     self.collect_data_from_file()

        # if mv grid district list is still unknown, get from results data
        # if (self.excluded_grid_districts is not None and
        #     self.nd is not None
        #     and mv_grid_districts is None):
        #     self.mv_grid_districs = [
        #         int(self.nd._mv_grid_districts[id].id_db)
        #         for id in list(range(0, self.nd._mv_grid_districts.__len__()))]
        #     self.mv_grid_districts_including_invalid = self.mv_grid_districs + \
        #         self.excluded_grid_districts

    def concat_nd_pickles(self, mv_grid_districts):
        """Read multiple pickles, join nd objects and save to file

        Parameters
        ----------
        mv_grid_districts : list
            Ints describing MV grid districts
        """

        pickle_name = cfg_dingo.get('output', 'nd_pickle')
        # self.nd = self.read_pickles_from_files(pickle_name)

        mvgd_1 = pickle.load(
            open(os.path.join(
                self.base_path,
                'results',
                pickle_name.format(mv_grid_districts[0])),
                'rb'))

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
                        "dingo_grids_{0}-{1}.pkl".format(
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


    def concat_csv_stats_files(self, ranges):
        """Concatenate multiple csv files containing statistics on nodes and edges.

        Parameters
        ----------
        ranges : list
            The list contains tuples of 2 elements describing start and end of each range.
        """

        for f in ['nodes', 'edges']:
            file_base_name = 'mvgd_' + f + '_stats_{0}-{1}.csv'

            filenames = []
            [filenames.append(file_base_name.format(mvgd_ids[0], mvgd_ids[1]))
             for mvgd_ids in ranges]

            results_file = 'mvgd_{0}_stats_{1}-{2}.csv'.format(
                f, ranges[0][0], ranges[-1][-1])

            self.concat_and_save_csv(filenames, results_file)


    def concat_and_save_csv(self, filenames, result_filename):
        """Concatenate and save multiple csv files in `base_path` specified by
        filnames

        The path specification of files in done via `self.base_path` in the
        `__init__` method of this class.

        Parameters
        ----------
        filenames : :any:`list` of :any:`str`
            Files to be concatenates
        result_filename : str
            File name of resulting file

        """

        list_ = []

        for filename in filenames:
            df = pd.read_csv(os.path.join(self.base_path, 'results', filename),
                             index_col=None, header=0)
            list_.append(df)

        frame = pd.concat(list_)
        frame.to_csv(os.path.join(
            self.base_path,
            'results', result_filename), index=False)

    def read_csv_results(self, concat_csv_file_range):
        """Read csv files (nodes and edges) containing results figures
        
        Parameters
        ----------
        concat_csv_file_range : :any:`list` of :any:`int`
            Ints describe first and last mv grid id
        """

        self.nodes = pd.read_csv(
            os.path.join(self.base_path,
                         'results',
                         'mvgd_nodes_stats_{0}-{1}.csv'.format(
                             concat_csv_file_range[0], concat_csv_file_range[-1]
                         ))
        )

        self.edges = pd.read_csv(
            os.path.join(self.base_path,
                         'results',
                         'mvgd_edges_stats_{0}-{1}.csv'.format(
                             concat_csv_file_range[0], concat_csv_file_range[-1]
                         ))
        )




    def global_stats(self):
        """#TODO docstring
        
        Returns
        -------
        :any:`int` 
            Description
        """
        if self.global_stats is None:
            self.global_stats = self.calculate_global_stats()

        return self.global_stats

    def mvdg_stats(self):
        """#TODO docstring
        
        Returns
        -------
        :any:`int` 
            Description
        """

        if self.mvgd_stats is None:
            self.mvgd_stats = self.calculate_mvgd_stats()

        return self.mvgd_stats

    def calculate_global_stats(self):
        """#TODO docstring
        
        Returns
        -------
        :any:`int` 
            Description
        """

        global_stats = {
            'Valid MV grid districts': "{0} out of {1}".format(
                len(self.mv_grid_districs),
                len(self.mv_grid_districts_including_invalid)
            )
        }

        return global_stats

    def calculate_mvgd_stats(self):
        """#TODO docstring
        
        Returns
        -------
        :any:`int` 
            Description
        """

        mvgd_stats = self.nodes.groupby('grid_id').sum()[
            ['peak_load', 'generation_capacity']] / 1e3
        mvgd_stats['v_nom'] = self.nodes.groupby('grid_id').mean()['v_nom']
        cable_line_km = self.edges['length'].groupby(
            [self.edges['grid_id'], self.edges['type_kind']]).sum().unstack(
            level=-1).fillna(0)
        cable_line_km.columns.name = None
        mvgd_stats[['km_cable', 'km_line']] = cable_line_km
        mvgd_stats['rings'] = self.nodes.groupby('grid_id').mean()['rings']
        type = self.nodes.groupby(['grid_id','type']).count()['node_id'].unstack(level=-1).fillna(0)
        type.columns.name= None
        mvgd_stats = pd.concat([mvgd_stats, type], axis=1)


        return mvgd_stats

    def plot_cable_length(self):
        """Cable length per MV grid district
        """

        stats = self.calculate_mvgd_stats().rename(columns={'v_nom': 'Voltage level'})

        # cable and line kilometer distribution
        f, axarr = plt.subplots(2, sharex=True)
        stats.hist(column=['km_cable'], bins=5, alpha=0.5, ax=axarr[0])
        stats.hist(column=['km_line'], bins=5, alpha=0.5, ax=axarr[1])

        plt.savefig(os.path.join(self.base_path, 'plots',
                                 'Histogram_cable_line_length.pdf'))

        # Generation capacity vs. peak load
        sns.set_context("paper", font_scale=1.1)
        sns.set_style("ticks")

        sns.lmplot('generation_capacity', 'peak_load',
                   data=stats,
                   fit_reg=False,
                   hue='Voltage level',
                   scatter_kws={"marker": "D",
                                "s": 100},
                   aspect=2)
        plt.title('Peak load vs. generation capcity')
        plt.xlabel('Generation capacity in MW')
        plt.ylabel('Peak load in MW')

        plt.savefig(os.path.join(self.base_path, 'plots',
                                 'Scatter_generation_load.pdf'))

        # Cable vs. line kilometer scatter
        sns.lmplot('km_cable', 'km_line',
                   data=stats,
                   fit_reg=False,
                   hue='Voltage level',
                   scatter_kws={"marker": "D",
                                "s": 100},
                   aspect=2)
        plt.title('Kilometer of cable/line')
        plt.xlabel('Km of cables')
        plt.ylabel('Km of overhead lines')

        plt.savefig(os.path.join(self.base_path, 'plots',
                                 'Scatter_cables_lines.pdf'))

def lv_grid_stats(nd):
    """Calculate statistics about LV grids

    Parameters
    ----------
    nd : LVGridDingo
        Network container object

    Returns
    -------
    :obj:`dict` 
        Dict with keys of LV grid repr() on first level. Each of the grids has a set of statistical information about its topology
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
