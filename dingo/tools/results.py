import pickle
import os
import pandas as pd

from dingo.tools import config as cfg_dingo

class ResultsDingo:
    """
    Holds raw results data and provides methods to generate a set of results

    `base_path` is optional, but it's actually required

    """

    def __init__(self, mv_grid_districts=None, filenames=None, base_path=''):
        """

        Either provide list of mv_grid_districts or dict of file names

        filenames : dict
            File names. Dictionary with certain keys
                * nd
                * edges
                * nodes

        `base_path` is optional, but it's actually required

        Parameters
        ----------
        """

        if mv_grid_districts is not None:
            self.mv_grid_districts_including_invalid = mv_grid_districts
        else:
            self.mv_grid_districts_including_invalid = None


        self.base_path = base_path

        self.nd = None
        self.edges = None
        self.nodes = None
        self.global_stats = None

        if os.path.isfile(os.path.join(self.base_path,
                         'info',
                          'corrupt_mv_grid_districts.txt')):
            self.invalid_mvgd = pd.read_csv(
                os.path.join(self.base_path,
                             'info',
                             'corrupt_mv_grid_districts.txt'))

        # get list of excluded grids (invalid) from info file
        self.excluded_grid_districts = self.invalid_mvgd['id'].tolist()

        # define currently valid mv_grid districts
        if self.mv_grid_districts_including_invalid is not None:
            self.mv_grid_districs = [
                mv for mv in self.mv_grid_districts_including_invalid
                if mv not in self.excluded_grid_districts]

        # read results data from a single file
        if filenames is not None:
            if 'nd' in list(filenames.keys()):
                self.nd = self.read_nd_multiple_mvgds(filenames['nd'])
            if 'edges' in list(filenames.keys()):
                self.edges = pd.read_csv(os.path.join(self.base_path,
                                                      'results',
                                                      filenames['edges']))
            if 'nodes' in list(filenames.keys()):
                self.nodes = pd.read_csv(os.path.join(self.base_path,
                                                      'results',
                                                      filenames['nodes']))
        # read results from single file per mv grid district
        elif mv_grid_districts is not None:
            self.collect_data_from_file()

        # if mv grid district list is still unknown, get from results data
        if (self.excluded_grid_districts is not None and
            self.nd is not None
            and mv_grid_districts is None):
            self.mv_grid_districs = [
                int(self.nd._mv_grid_districts[id].id_db)
                for id in list(range(0, self.nd._mv_grid_districts.__len__()))]
            self.mv_grid_districts_including_invalid = self.mv_grid_districs + \
                self.excluded_grid_districts

    def read_pickles_from_files(self, pickle_name):
        """
        Read multiple pickles and join nd objects

        :param pickle_name:
        :return:
        """
        mvgd_1 = pickle.load(
            open(os.path.join(
                self.base_path,
                'results',
                pickle_name.format(self.mv_grid_districs[0])),
                'rb'))

        for mvgd in self.mv_grid_districs[1:]:
            mvgd_pickle = pickle.load(open(os.path.join(
                self.base_path,
                'results', pickle_name.format(mvgd)), 'rb'))
            mvgd_1.add_mv_grid_district(mvgd_pickle._mv_grid_districts[0])

        return mvgd_1

    def read_csv_from_files(self, csv_name):
        """
        Read multiple CSV files and concatenate these
        :param csv_name:
        :return:
        """
        mvgd_1 = pd.read_csv(
            os.path.join(
                self.base_path,
                'results',
                csv_name.format(self.mv_grid_districs[0])))

        for mvgd in self.mv_grid_districs[1:]:
            mvgd_df = pd.read_csv(os.path.join(
                self.base_path,
                'results', csv_name.format(mvgd)))
            mvgd_1 = mvgd_1.append(mvgd_df, ignore_index=True)

        return mvgd_1

    def collect_data_from_file(self):
        """
        Read results data from multiple files

        :return:
        """

        # load nd object from pickle
        pickle_name = cfg_dingo.get('output', 'nd_pickle')
        self.nd = self.read_pickles_from_files(pickle_name)

        # load nodes and edges table data
        edges = cfg_dingo.get('output', 'edges_stats')
        self.edges = self.read_csv_from_files(edges)
        self.edges['grid_id'] = self.edges['grid_id'].map(
            lambda x: '%.0f' % x)
        nodes = cfg_dingo.get('output', 'nodes_stats')
        self.nodes = self.read_csv_from_files(nodes)
        self.nodes['grid_id'] = self.nodes['grid_id'].map(
            lambda x: '%.0f' % x)

    def read_nd_multiple_mvgds(self, filename):
        """
        Reads file of multiple grid district stored in base path

        Parameters
        ----------
        filename : str
            Filename.
            File must be saved in `base_path` structure.
        """
        data = pickle.load(
            open(os.path.join(
                self.base_path,
                'results',
                filename),
                'rb'))
        return data

    def save_merge_data(self):
        """Save complete dataset of a run"""
        pickle.dump(self.nd,
                open(os.path.join(
                    self.base_path,
                    'results',
                    "dingo_grids_{0}-{1}.pkl".format(
                    self.mv_grid_districs[0], self.mv_grid_districs[-1])),
                     "wb"))
        self.nodes.to_csv(os.path.join(
                    self.base_path,
                    'results', 'mvgd_nodes_stats_{0}-{1}.csv'.format(
                    self.mv_grid_districs[0], self.mv_grid_districs[-1])),
            index=False
        )
        self.edges.to_csv(os.path.join(
                    self.base_path,
                    'results', 'mvgd_edges_stats_{0}-{1}.csv'.format(
                    self.mv_grid_districs[0], self.mv_grid_districs[-1])),
            index=False
        )

    def global_stats(self):

        if self.global_stats is None:
            self.global_stats = self.calculate_global_stats()

        return self.global_stats

    def calculate_global_stats(self):

        global_stats = {
            'Valid MV grid districts': "{0} out of {1}".format(
                len(self.mv_grid_districs),
                len(self.mv_grid_districts_including_invalid)
            )
        }

        return global_stats

    # TODO: add load for each not to nodes table
