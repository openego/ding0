#from dingo.core.network import GridDingo
from . import GridDingo
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo, CircuitBreakerDingo
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
    default_branch_kind: kind of branch (possible values: 'cable' or 'line')
    default_branch_type: type of branch (pandas Series object with cable/line parameters)
    """
    # TODO: Add method to join MV graph with LV graphs to have one graph that covers whole grid (MV and LV)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        #more params
        self._station = None
        self._cable_distributors = []
        self._circuit_breakers = []
        self.default_branch_kind = kwargs.get('default_branch_kind', None)
        self.default_branch_type = kwargs.get('default_branch_type', None)
        self.default_branch_type_aggregated = kwargs.get('default_branch_type_aggregated', None)

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

    def circuit_breakers(self):
        """Returns a generator for iterating over circuit breakers"""
        for circ_breaker in self._circuit_breakers:
            yield circ_breaker

    def circuit_breakers_count(self):
        """Returns the count of circuit breakers in MV grid"""
        return len(self._circuit_breakers)

    def add_circuit_breaker(self, circ_breaker):
        """ Creates circuit breaker object and ...

        Args:
            circ_breaker: CircuitBreakerDingo object
        """
        if circ_breaker not in self._circuit_breakers and isinstance(circ_breaker, CircuitBreakerDingo):
            self._circuit_breakers.append(circ_breaker)
            self.graph_add_node(circ_breaker)

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
        self._graph = mv_connect.mv_connect(self, self._graph, LVLoadAreaCentreDingo(), debug)

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

        # set default branch type
        self.default_branch_type, self.default_branch_type_aggregated = self.set_default_branch_type(debug)

        # choose appropriate transformers for each MV sub-station
        self._station.choose_transformers()

        # choose appropriate type of line/cable for each edge
        # TODO: move line parametrization to routing process
        #self.parametrize_lines()

    def set_default_branch_type(self, debug=False):
        """ Determines default branch type according to grid district's peak load and standard equipment.

        Args:
            debug: If True, information is printed during process

        Returns:
            default branch type (pandas Series object). If no appropriate type is found, return largest possible one.

        Notes
        -----
        Parameter values for cables and lines are taken from [1]_, [2]_ and [3]_.

        Lines are chosen to have 60 % load relative to their nominal capacity according to [4]_.

        Decision on usage of overhead lines vs. cables is determined by load density of the considered region. Urban
        areas usually are equipped with underground cables whereas rural areas often have overhead lines as MV
        distribution system [5]_.

        References
        ----------
        .. [1] Klaus Heuck et al., "Elektrische Energieversorgung", Vieweg+Teubner, Wiesbaden, 2007
        .. [2] René Flosdorff et al., "Elektrische Energieverteilung", Vieweg+Teubner, 2005
        .. [3] Helmut Alt, "Vorlesung Elektrische Energieerzeugung und -verteilung"
            http://www.alt.fh-aachen.de/downloads//Vorlesung%20EV/Hilfsb%2044%20Netzdaten%20Leitung%20Kabel.pdf, 2010
        .. [4] Deutsche Energie-Agentur GmbH (dena), "dena-Verteilnetzstudie. Ausbau- und Innovationsbedarf der
            Stromverteilnetze in Deutschland bis 2030.", 2012
        .. [5] Tao, X., "Automatisierte Grundsatzplanung von
            Mittelspannungsnetzen", Dissertation, RWTH Aachen, 2007
        """

        package_path = dingo.__path__[0]

        # decide whether cable or line is used (initially for entire grid) and set grid's attribute
        if self.v_level == 20:
            self.default_branch_kind = 'line'
        elif self.v_level == 10:
            self.default_branch_kind = 'cable'

        # get max. count of half rings per MV grid district
        mv_half_ring_count_max = int(cfg_dingo.get('mv_routing_tech_constraints',
                                                   'mv_half_ring_count_max'))

        # load cable/line assumptions, file_names and parameter
        if self.default_branch_kind == 'line':
            load_factor_normal = float(cfg_dingo.get('assumptions',
                                                     'load_factor_line_normal'))
            equipment_parameters_file = cfg_dingo.get('equipment',
                                                      'equipment_parameters_lines')
            branch_parameters = pd.read_csv(os.path.join(package_path, 'data',
                                            equipment_parameters_file),
                                            comment='#',
                                            converters={'I_max_th': lambda x: int(x), 'U_n': lambda x: int(x)})

        elif self.default_branch_kind == 'cable':
            load_factor_normal = float(cfg_dingo.get('assumptions',
                                                     'load_factor_cable_normal'))
            equipment_parameters_file = cfg_dingo.get('equipment',
                                                      'equipment_parameters_cables')
            branch_parameters = pd.read_csv(os.path.join(package_path, 'data',
                                            equipment_parameters_file),
                                            comment='#',
                                            converters={'I_max_th': lambda x: int(x), 'U_n': lambda x: int(x)})
        else:
            raise ValueError('Grid\'s default_branch_kind is invalid, could not set branch parameters.')

        # select appropriate branch params according to voltage level, sorted ascending by max. current
        branch_parameters = branch_parameters[branch_parameters['U_n'] == self.v_level].sort_values('I_max_th')

        # get largest line/cable type
        branch_type_max = branch_parameters.loc[branch_parameters['I_max_th'].idxmax()]

        # set aggregation flag using largest available line/cable
        self.set_nodes_aggregation_flag(branch_type_max['I_max_th'] * load_factor_normal)

        # calc peak current sum (= "virtual" current) of whole grid (I = S * sqrt(3) / U) excluding load areas of type
        # satellite and aggregated
        peak_current_sum = ((self.grid_district.peak_load -
                             self.grid_district.peak_load_satellites -
                             self.grid_district.peak_load_aggregated) *
                            (3**0.5) / self.v_level)  # units: kVA / kV = A

        # search the smallest possible line/cable for MV grid district in equipment datasets for all load areas
        # excluding those of type satellite and aggregated
        for idx, row in branch_parameters.iterrows():
            # calc number of required rings using peak current sum of grid district,
            # load factor and max. current of line/cable
            half_ring_count = round(peak_current_sum / (row['I_max_th'] * load_factor_normal))

            if debug:
                print('=== Selection of default branch type in', self, '===')
                print('Peak load=', self.grid_district.peak_load, 'kVA')
                print('Peak current=', peak_current_sum)
                print('I_max_th=', row['I_max_th'])
                print('Half ring count=', half_ring_count)

            # if count of half rings is below or equal max. allowed count, use current branch type as default
            if half_ring_count <= mv_half_ring_count_max:
                return row, branch_type_max

        # no equipment was found, return largest available line/cable

        if debug:
            print('No appropriate line/cable type could be found for', self, ', declare some load areas as aggregated.')

        return branch_type_max, branch_type_max

    def set_nodes_aggregation_flag(self, peak_current_branch_max):
        """ Set LV load areas with too high demand to aggregated type.

        Args:
            peak_current_branch_max: Max. allowed current for line/cable

        Returns:
            nothing
        """

        for lv_load_area in self.grid_district.lv_load_areas():
            peak_current_node = (lv_load_area.peak_load_sum * (3**0.5) / self.v_level)  # units: kVA / kV = A
            if peak_current_node > peak_current_branch_max:
                lv_load_area.is_aggregated = True

        # add peak demand for all LV load areas of aggregation type
        self.grid_district.add_aggregated_peak_demand()

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
        Selects typified model grid based on population

        Parameters
        ----------
        string_properties: DataFrame
            Properties of LV typified model grids
        apartment_string: DataFrame
            Relational table of apartment count and strings of model grid
        apartment_trafo: DataFrame
            Relational table of apartment count and trafo size
        population: Int
            Population within LV grid district

        Returns
        -------
        selected_strings_df: DataFrame
            Selected string of typified model grid
        transformer: Int
            Size of Transformer given in kVar
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

        transformer = apartment_trafo.loc[apartments]

        return selected_strings_df, transformer


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