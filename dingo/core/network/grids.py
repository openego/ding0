#from dingo.core.network import GridDingo
from . import GridDingo
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo
from dingo.core.network import CableDistributorDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo
from dingo.grid.mv_grid import mv_routing
from dingo.grid.mv_grid import mv_connect
import dingo
from dingo.tools import config as cfg_dingo

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

    def set_voltage_level(self):
        """ Sets voltage level of MV grid according to load density.

        Args:
            none
        Returns:
            nothing

        Notes
        -----
        Decision on voltage level is determined by load density of the considered region. Urban areas (load density of
        >= 1 MW/km2 according to [1]_) usually got a voltage of 10 kV whereas rural areas mostly use 20 kV.

        References
        ----------
        .. [1] Falk Schaller et al., "Modellierung realitätsnaher zukünftiger Referenznetze im Verteilnetzsektor zur
            Überprüfung der Elektroenergiequalität", Internationaler ETG-Kongress Würzburg, 2011
        """
        # TODO: more references!

        load_density_threshold = float(cfg_dingo.get('assumptions',
                                                     'load_density_threshold'))

        # calculate load density
        # TODO: Move constant 1e6 to config file
        load_density = ((self.grid_district.peak_load / 1e3) /
                        (self.grid_district.geo_data.area / 1e6)) # unit MVA/km^2

        # identify voltage level
        if load_density < load_density_threshold:
            self.v_level = 20
        elif load_density >= load_density_threshold:
            self.v_level = 10
        else:
            raise ValueError('load_density is invalid!')

    def parametrize_grid(self, debug=False):
        """ Performs Parametrization of grid equipment.

        Args:
            debug: If True, information is printed during process
        """
        # TODO: Add more detailed description
        # TODO: Pass debug flag to functions

        # set grid's voltage level
        self.set_voltage_level()

        # set MV station's voltage level
        self._station.set_operation_voltage_level()

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

        self._stations = []

    def stations(self):
        """Returns a generator for iterating over LV stations"""
        for station in self._stations:
            yield station

    def add_station(self, lv_station):
        """Adds a LV station to _stations and grid graph if not already existing"""
        if lv_station not in self.stations() and isinstance(lv_station, LVStationDingo):
            self._stations.append(lv_station)

            self.graph_add_node(lv_station)
            self.grid_district.mv_grid_district.mv_grid.graph_add_node(lv_station)

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