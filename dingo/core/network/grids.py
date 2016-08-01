#from dingo.core.network import GridDingo
from . import GridDingo
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo
from dingo.core.network import CableDistributorDingo, LVLoadDingo, \
LVCableDistributorDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo
from dingo.grid.mv_grid import mv_routing
from dingo.grid.mv_grid import mv_connect
import dingo
from dingo.tools import config as cfg_dingo
import dingo.core

import networkx as nx
import pandas as pd
import os


class MVGridDingo(GridDingo):
    """ DINGO medium voltage grid

    Parameters
    ----------
    region : MV region (instance of MVGridDistrictDingo class) that is associated with grid
    """
    # TODO: Add method to join MV graph with LV graphs to have one graph that covers whole grid (MV and LV)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #more params
        self._station = None
        self._cable_distributors = []

        self.add_station(kwargs.get('station', None))

    def station(self):
        """Returns MV station"""
        return self._station

    def cable_distributors(self):
        """Returns a generator for iterating over cable distributors"""
        for cable_dist in self._cable_distributors:
            yield cable_dist

    def cable_distributors_count(self):
        """Returns the count of cable distributors in MV grid"""
        return len(self._cable_distributors)

    def add_station(self, mv_station, force=False):
        """Adds MV station if not already existing

        mv_station: MVStationDingo object
        force: bool. If True, MV Station is set even though it's not empty (override)
        """
        # TODO: Use this exception handling as template for similar methods in other classes
        if not isinstance(mv_station, MVStationDingo):
            raise Exception('Given MV station is not a MVStationDingo object.')
        if self._station is None:
            self._station = mv_station
            self.graph_add_node(mv_station)
        else:
            if force:
                self._station = mv_station
            else:
                raise Exception('MV Station already set, use argument `force=True` to override.')

    # TODO: Following code builds graph after all objects are added (called manually) - maybe used later instead of ad-hoc adding
    # def graph_build(self):
    #     """Builds/fills graph with objects (stations, ..)"""
    #     # TODO: Add edges, loads etc. later on
    #
    #     # add MV station
    #     self.graph_add_node(self._station)
    #
    #     # add LV stations
    #     # TODO: to get LV stations, generator of generators is necessary
    #     # TODO: see http://stackoverflow.com/questions/19033401/python-generator-of-generators
    #     # TODO: not sure if the following works:
    #     for lv_station in [grid.stations() for grid in [region.lv_grids() for region in self.region.lv_load_areas()]]:
    #         self.graph_add_node(lv_station)

    def add_cable_distributor(self, cable_dist):
        """Adds a cable distributor to _cable_distributors if not already existing"""
        if cable_dist not in self.cable_distributors() and isinstance(cable_dist, CableDistributorDingo):
            # add to array and graph
            self._cable_distributors.append(cable_dist)
            self.graph_add_node(cable_dist)

    def routing(self, debug=False, anim=None):
        """ Performs routing on grid graph nodes, adds resulting edges

        Args:
            debug: If True, information is printed while routing
        """

        # do the routing
        self._graph = mv_routing.solve(self._graph, debug, anim)
        self._graph = mv_connect.mv_connect(self._graph, LVLoadAreaCentreDingo(), debug)

        # create MV Branch objects from graph edges (lines) and link these objects back to graph edges
        # TODO:
        # mv_branches = {}
        # for edge in self._graph.edges():
        #     mv_branch = BranchDingo()
        #     mv_branches[edge] = mv_branch
        # nx.set_edge_attributes(self._graph, 'branch', mv_branches)

    def parametrize_grid(self, debug=False):
        """ Performs Parametrization of grid equipment.

        Args:
            debug: If True, information is printed while routing
        """
        # TODO: Add more detailed description
        # TODO: Pass debug flag to functions

        # choose appropriate transformers for each MV sub-station
        self._station.choose_transformers()

        # choose appropriate type of line/cable for each edge
        # TODO: move line parametrization to routing process
        #self.parametrize_lines()

    def parametrize_lines(self):
        """Chooses line/cable type and defines parameters

        Adds relevant parameters to medium voltage lines of routed grids. It is
        assumed that for each MV circuit same type of overhead lines/cables are
        used. Furthermore, within each circuit no mix of overhead lines/ cables
        is applied.

        Notes
        -----
        Parameter values are take from [1]_.

        Lines are chosen to have 60 % load relative to their nominal capacity
        according to [2]_.

        Decision on usage of overhead lines vs. cables is determined by load
        density of the considered region. Urban areas (load density of
        >= 1 MW/km2 according to [3]_) usually are equipped with underground
        cables whereas rural areas often have overhead lines as MV distribution
        system [4]_.

        References
        ----------
        .. [1] Helmut Alt, "Vorlesung Elektrische Energieerzeugung und
            -verteilung"
            http://www.alt.fh-aachen.de/downloads//Vorlesung%20EV/Hilfsb%2044%
            20Netzdaten%20Leitung%20Kabel.pdf, 2010
        .. [2] Deutsche Energie-Agentur GmbH (dena), "dena-Verteilnetzstudie.
            Ausbau- und Innovationsbedarf der Stromverteilnetze in Deutschland
            bis 2030.", 2012
        .. [3] Falk Schaller et al., "Modellierung realitätsnaher zukünftiger
            Referenznetze im Verteilnetzsektor zur Überprüfung der
            Elektroenergiequalität", Internationaler ETG-Kongress Würzburg, 2011
        .. [4] Tao, X., "Automatisierte Grundsatzplanung von
            Mittelspannungsnetzen", Dissertation, RWTH Aachen, 2007
        """

        # load assumptions
        load_density_threshold= float(cfg_dingo.get('assumptions',
                                                    'load_density_threshold'))
        load_factor_line = float(cfg_dingo.get('assumptions',
                                               'load_factor_line'))
        load_factor_cable = float(cfg_dingo.get('assumptions',
                                                'load_factor_cable'))

        # load cable/line parameters (after loading corresponding file names)
        package_path = dingo.__path__[0]
        equipment_parameters_lines = cfg_dingo.get('equipment',
                                                   'equipment_parameters_lines')
        equipment_parameters_cables = cfg_dingo.get('equipment',
                                                    'equipment_parameters_cables')

        line_parameter = pd.read_csv(os.path.join(package_path, 'data',
                                     equipment_parameters_lines),
                                     converters={'i_max_th': lambda x: int(x)})
        cable_parameter = pd.read_csv(os.path.join(package_path, 'data',
                                      equipment_parameters_cables),
                                      converters={'I_n': lambda x: int(x)})


        # iterate over edges (lines) of graph
        for edge in self.graph_edges():

            # calculate load density
            # TODO: Move constant 1e6 to config file
            load_density = ((self.region.peak_load / 1e3) /
                            (self.region.geo_data.area / 1e6)) # unit MVA/km^2

            # identify voltage level
            # identify type: line or cable
            # TODO: is this simple approach valuable?
            # see: dena Verteilnetzstudie
            if load_density < load_density_threshold:
                edge['branch'].v_level = 20
                branch_type = 'cable'
            elif load_density >= load_density_threshold:
                edge['branch'].v_level = 10
                branch_type = 'line'
            else:
                raise ValueError('load_density has to be greater than 0!')

            peak_current = self.region.peak_load / edge['branch'].v_level

            # choose line/cable type according to peak load of mv_grid
            if branch_type is 'line':
                # TODO: cross-check is multiplication by 3 is right
                line_name = line_parameter.ix[line_parameter[
                    line_parameter['i_max_th'] * 3 * load_factor_line >= peak_current]
                ['i_max_th'].idxmin()]['name']

                # set parameters to branch object
                edge['branch'].x = float(line_parameter.loc[
                                        line_parameter['name'] == line_name, 'x'])
                edge['branch'].r = float(line_parameter.loc[
                                            line_parameter['name'] == line_name, 'r'])
                edge['branch'].i_max_th = float(line_parameter.loc[
                                                   line_parameter['name'] == line_name, 'i_max_th'])
                edge['branch'].type = branch_type
            elif branch_type is 'cable':
                cable_name = cable_parameter.ix[cable_parameter[
                    cable_parameter['I_n'] * 3 * load_factor_cable >= peak_current]
                ['I_n'].idxmin()]['name']

                # set parameters to branch object
                edge['branch'].x = float(cable_parameter.loc[
                                            cable_parameter['name'] == cable_name, 'x_L'])
                edge['branch'].r = float(cable_parameter.loc[
                                            cable_parameter['name'] == cable_name, 'r'])
                edge['branch'].i_max_th = float(cable_parameter.loc[
                                                   cable_parameter['name'] == cable_name, 'I_n'])
                edge['branch'].type = branch_type

    def __repr__(self):
        return 'mv_grid_' + str(self.id_db)


class LVGridDingo(GridDingo):
    """ DINGO low voltage grid

    Parameters
    ----------
    region : LV region (instance of LVLoadAreaDingo class) that is associated with grid
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._station = []
        self._loads = []
        self._cable_dists = []
        self.population = kwargs.get('population', None)

    def station(self):
        """Returns a generator for iterating over LV station"""
        for station in self._station:
            yield station

    def add_station(self, lv_station):
        """Adds a LV station to _station and grid graph if not already existing"""
        if lv_station not in self.station() and isinstance(lv_station, LVStationDingo):
            self._station.append(lv_station)
            self.graph_add_node(lv_station)

    def add_load(self, lv_load):
        """Adds a LV load to _loads and grid graph if not already existing"""
        if lv_load not in self._loads and isinstance(lv_load,
                                                           LVLoadDingo):
            self._loads.append(lv_load)
            self._graph.add_node(lv_load)

    def add_cable_dist(self, lv_cable_dist):
        """Adds a LV cable_dist to _cable_dists and grid graph if not already existing"""
        if lv_cable_dist not in self._cable_dists and isinstance(lv_cable_dist,
                                                       LVCableDistributorDingo):
            self._cable_dists.append(lv_cable_dist)
            self._graph.add_node(lv_cable_dist)

    def cable_dists(self):
        """Returns a generator for iterating over LV _cable_dist"""
        for cable_dist in self._cable_dists:
            yield cable_dist

    def loads(self):
        """Returns a generator for iterating over LV _load"""
        for load in self._loads:
            yield load

    def select_typified_grid_model(self,
                                   string_properties,
                                   apartment_string,
                                   apartment_trafo,
                                   population):
        """

        Parameters
        ----------
        string_properties:
        apartment_trafo:
        apartment_string:
        population:

        """

        apartment_house_branch_ratio = cfg_dingo.get("assumptions",
            "apartment_house_branch_ratio")
        population_per_apartment = cfg_dingo.get("assumptions",
            "population_per_apartment")

        apartments = round(population / population_per_apartment)
        if apartments > 196:
            apartments = 196

        # select set of strings that represent one type of model grid
        strings = apartment_string.loc[apartments]
        selected_strings = [int(s) for s in strings[strings >= 1].index.tolist()]

        # slice dataframe of string parameters
        selected_strings_df = string_properties.loc[selected_strings]

        # add number of occurences of each branch to df
        occurence_selector = [str(i) for i in selected_strings]
        selected_strings_df['occurence'] = strings.loc[occurence_selector].tolist()

        return selected_strings_df


    def build_lv_graph(self, selected_string_df):
        """
        Builds nxGraph based on the LV grid model

        Parameter
        ---------
        selected_string_df: Dataframe
            Table of strings of the selected grid model

        Notes
        -----
        To understand what is happening in this method a few data table columns
        are explained here

        * `count house branch`: number of houses connected to a string
        * `distance house branch`: distance on a string between two house
            branches
        * `string length`: total length of a string
        * `length house branch A|B`: cable from string to connection point of a
            house

        A|B in general brings some variation in to the typified model grid and
        refer to different length of house branches and different cable types
        respectively different cable widths.
        """

        # iterate over each type of branch
        for i, row in selected_string_df.iterrows():
            # iterate over it's occurences
            for branch_no in range(1, int(row['occurence']) + 1):
                # iterate over house branches
                for house_branch in range(1, row['count house branch'] + 1):
                    if house_branch % 2 == 0:
                        variant = 'B'
                    else:
                        variant = 'A'
                    lv_cable_dist = LVCableDistributorDingo(
                        id=self.grid_district.id_db,
                        string_id=i,
                        branch_no=branch_no,
                        load_no=house_branch)

                    lv_load = LVLoadDingo(id=self.grid_district.id_db,
                                          string_id=i,
                                          branch_no=branch_no,
                                          load_no=house_branch)

                    # add lv_load and lv_cable_dist to graph
                    self.add_load(lv_load)
                    self.add_cable_dist(lv_cable_dist)

                    cable_name = row['cable type'] + \
                                       ' 4x1x{}'.format(row['cable width'])

                    # connect current lv_cable_dist to last one
                    if house_branch == 1:
                        # edge connect first house branch in branch with the station
                        self._graph.add_edge(
                            self._station[0],
                            lv_cable_dist,
                            branch=BranchDingo(
                                length=row['distance house branch'],
                                type=cable_name
                                ))
                    else:
                        self._graph.add_edge(
                            self._cable_dists[-2],
                            lv_cable_dist,
                            branch=BranchDingo(
                                length=row['distance house branch'],
                                type=cable_name))

                    # connect house to cable distributor
                    house_cable_name = row['cable type {}'.format(variant)] + \
                        ' 4x1x{}'.format(row['cable width {}'.format(variant)])
                    self._graph.add_edge(
                        lv_cable_dist,
                        lv_load,
                        branch=BranchDingo(
                            length=row['length house branch {}'.format(
                                variant)],
                            type=dingo.core.lv_cable_parameters. \
                                loc[house_cable_name]))

    # TODO: Following code builds graph after all objects are added (called manually) - maybe used later instead of ad-hoc adding
    # def graph_build(self):
    #     """Builds/fills graph with objects (stations, ..)"""
    #     # TODO: Add edges, loads etc. later on
    #
    #     # add LV stations
    #     for lv_station in self.stations():
    #         self.graph_add_node(lv_station)
    #
    #     # TODO: add more nodes (loads etc.) here

    def __repr__(self):
        return 'lv_grid_' + str(self.id_db)