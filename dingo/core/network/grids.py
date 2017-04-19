# from dingo.core.network import GridDingo
from . import GridDingo
from dingo.core.network.stations import *
from dingo.core.network import RingDingo, BranchDingo, CircuitBreakerDingo
from dingo.core.network.loads import *
from dingo.core import MVCableDistributorDingo
from dingo.core.network.cable_distributors import LVCableDistributorDingo
from dingo.grid.mv_grid import mv_routing
from dingo.grid.mv_grid import mv_connect
import dingo
from dingo.tools import config as cfg_dingo, pypsa_io, tools
from dingo.tools import config as cfg_dingo, tools
from dingo.grid.mv_grid.tools import set_circuit_breakers
from dingo.flexopt.reinforce_grid import *
import dingo.core
from dingo.core.structure.regions import LVLoadAreaCentreDingo

import networkx as nx
import pandas as pd
import os
from datetime import datetime
from shapely.ops import transform
import pyproj
from functools import partial
import logging


logger = logging.getLogger('dingo')


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

        # more params
        self._station = None
        self._rings = []
        self._circuit_breakers = []
        self.default_branch_kind = kwargs.get('default_branch_kind', None)
        self.default_branch_type = kwargs.get('default_branch_type', None)
        self.default_branch_kind_settle = kwargs.get('default_branch_kind_settle', None)
        self.default_branch_type_settle = kwargs.get('default_branch_type_settle', None)
        self.default_branch_kind_aggregated = kwargs.get('default_branch_kind_aggregated', None)
        self.default_branch_type_aggregated = kwargs.get('default_branch_type_aggregated', None)

        self.add_station(kwargs.get('station', None))

    def station(self):
        """Returns MV station"""
        return self._station

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

    def open_circuit_breakers(self):
        """ Opens all circuit breakers in MV grid """
        for circ_breaker in self.circuit_breakers():
            circ_breaker.open()

    def close_circuit_breakers(self):
        """ Closes all circuit breakers in MV grid """
        for circ_breaker in self.circuit_breakers():
            circ_breaker.close()

    def add_station(self, mv_station, force=False):
        """ Adds MV station if not already existing

        Args:
            mv_station: MVStationDingo object
            force: bool. If True, MV Station is set even though it's not empty (override)
        """
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

    def add_load(self, lv_load):
        """Adds a MV load to _loads and grid graph if not already existing"""
        if lv_load not in self._loads and isinstance(lv_load,
                                                     MVLoadDingo):
            self._loads.append(lv_load)
            self.graph_add_node(lv_load)

    def add_cable_distributor(self, cable_dist):
        """Adds a cable distributor to _cable_distributors if not already existing"""
        if cable_dist not in self.cable_distributors() and isinstance(cable_dist,
                                                                      MVCableDistributorDingo):
            # add to array and graph
            self._cable_distributors.append(cable_dist)
            self.graph_add_node(cable_dist)

    def add_ring(self, ring):
        """Adds a ring to _rings if not already existing"""
        if ring not in self._rings and isinstance(ring, RingDingo):
            self._rings.append(ring)

    def rings_count(self):
        """Returns the count of rings in MV grid"""
        return len(self._rings)

    def rings_nodes(self, include_root_node=False, include_satellites=False):
        """ Returns a generator for iterating over rings (=routes of MVGrid's graph)

        Args:
            include_root_node: If True, the root node is included in the list of ring nodes.
            include_satellites: If True, the satellite nodes (nodes that diverge from ring nodes) is included in the
                                list of ring nodes.
        Returns:
            List with nodes of each ring of _graph in- or excluding root node (HV/MV station) (arg `include_root_node`),
            format: [ring_m_node_1, ..., ring_m_node_n]
        Notes:
            Circuit breakers must be closed to find rings, this is done automatically.
        """
        for circ_breaker in self.circuit_breakers():
            if circ_breaker.status is 'open':
                circ_breaker.close()
                logger.info('Circuit breakers were closed in order to find MV '
                            'rings')

        for ring in nx.cycle_basis(self._graph, root=self._station):
            if not include_root_node:
                ring.remove(self._station)

            if include_satellites:
                ring_nodes = ring
                satellites = []
                if include_root_node:
                    ring_nodes.remove(self._station)
                for ring_node in ring:
                    # determine all branches diverging from each ring node
                    satellites.append(self.graph_nodes_from_subtree(ring_node))
                # return ring and satellite nodes (flatted list of lists)
                yield ring + [_ for sublist in satellites for _ in sublist]
            else:
                yield ring

    def get_ring_from_node(self, node):
        """ Determines the ring (RingDingo object) which node is member of.
        Args:
            node: Dingo object (member of graph)
        Returns:
            RingDingo object
        """
        try:
            return self.graph_branches_from_node(node)[0][1]['branch'].ring
        except:
            raise Exception('Cannot get node\'s associated ring.')

    def graph_nodes_from_subtree(self, node_source):
        """ Finds all nodes of a tree that is connected to `node_source` and are (except `node_source`) not part of the
            ring of `node_source` (traversal of graph from `node_source` excluding nodes along ring).
            Example: A given graph with ring (edges) 0-1-2-3-4-5-0 and a tree starting at node (`node_source`) 3 with
            edges 3-6-7, 3-6-8-9 will return [6,7,8,9]

        Args:
            node_source: source node (Dingo object), member of _graph

        Returns:
            List of nodes (Dingo objects)
        """
        if node_source in self._graph.nodes():

            # get all nodes that are member of a ring
            for ring in self.rings_nodes(include_root_node=False):
                if node_source in ring:
                    node_ring = ring
                    break

            # result set
            nodes_subtree = set()

            # get nodes from subtree
            if node_source in node_ring:
                for path in nx.shortest_path(self._graph, node_source).values():
                    if len(path)>1:
                        if (path[1] not in node_ring) and (path[1] is not self.station()):
                            nodes_subtree.update(path[1:len(path)])
            else:
                raise ValueError(node_source, 'is not member of ring.')

        else:
            raise ValueError(node_source, 'is not member of graph.')

        return list(nodes_subtree)

    def set_branch_ids(self):
        """ Generates and sets ids of branches for MV and underlying LV grids """

        # MV grid:
        ctr = 1
        for branch in self.graph_edges():
            branch['branch'].id_db = self.grid_district.id_db * 10**4 + ctr
            ctr += 1

        # LV grid:
        for lv_load_area in self.grid_district.lv_load_areas():
            for lv_grid_district in lv_load_area.lv_grid_districts():
                ctr = 1
                for branch in lv_grid_district.lv_grid.graph_edges():
                    branch['branch'].id_db = lv_grid_district.id_db * 10**7 + ctr
                    ctr += 1

    def routing(self, debug=False, anim=None):
        """ Performs routing on grid graph nodes

        Args:
            debug: If True, information is printed while routing
        """

        # do the routing
        self._graph = mv_routing.solve(graph=self._graph,
                                       debug=debug,
                                       anim=anim)
        logger.info('==> MV Routing for {} done'.format(repr(self)))

        # connect satellites (step 1, with restrictions like max. string length, max peak load per string)
        self._graph = mv_connect.mv_connect_satellites(mv_grid=self,
                                                       graph=self._graph,
                                                       mode='normal',
                                                       debug=debug)
        logger.info('==> MV Sat1 for {} done'.format(repr(self)))

        # connect satellites to closest line/station on a MV ring that have not been connected in step 1
        self._graph = mv_connect.mv_connect_satellites(mv_grid=self,
                                                       graph=self._graph,
                                                       mode='isolated',
                                                       debug=debug)
        logger.info('==> MV Sat2 for {} done'.format(repr(self)))

        # connect stations
        self._graph = mv_connect.mv_connect_stations(mv_grid_district=self.grid_district,
                                                     graph=self._graph,
                                                     debug=debug)
        logger.info('==> MV Stations for {} done'.format(repr(self)))

    def connect_generators(self, debug=False):
        """ Connects MV generators (graph nodes) to grid (graph)

        Args:
            debug: If True, information is printed during process
        """

        self._graph = mv_connect.mv_connect_generators(self.grid_district, self._graph, debug)

    def parametrize_grid(self, debug=False):
        """ Performs Parametrization of grid equipment: 1. Sets voltage level of MV grid, 2. Operation voltage level
            and transformer of HV/MV station, 3. Default branch types (normal, aggregated, settlement)

        Args:
            debug: If True, information is printed during process
        Notes:
            It is assumed that only cables are used within settlements
        """
        # TODO: Add more detailed description

        # set grid's voltage level
        self.set_voltage_level()

        # set MV station's voltage level
        self._station.set_operation_voltage_level()

        # set default branch types (normal, aggregated areas and within settlements)
        self.default_branch_type,\
        self.default_branch_type_aggregated,\
        self.default_branch_type_settle = self.set_default_branch_type(debug)

        # set default branch kinds
        self.default_branch_kind_aggregated = self.default_branch_kind
        self.default_branch_kind_settle = 'cable'

        # choose appropriate transformers for each HV/MV sub-station
        self._station.choose_transformers()

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

        # transform MVGD's area to epsg 3035
        # to achieve correct area calculation
        projection = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:4326'),  # source coordinate system
            pyproj.Proj(init='epsg:3035'))  # destination coordinate system

        # calculate load density
        # TODO: Move constant 1e6 to config file
        load_density = ((self.grid_district.peak_load / 1e3) /
                        (transform(projection, self.grid_district.geo_data).area / 1e6)) # unit MVA/km^2

        # identify voltage level
        if load_density < load_density_threshold:
            self.v_level = 20
        elif load_density >= load_density_threshold:
            self.v_level = 10
        else:
            raise ValueError('load_density is invalid!')

    def set_default_branch_type(self, debug=False):
        """ Determines default branch type according to grid district's peak load and standard equipment.

        Args:
            debug: If True, information is printed during process

        Returns:
            default branch type: pandas Series object. If no appropriate type is found, return largest possible one.
            default branch type max: pandas Series object. Largest available line/cable type

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
        # TODO: [3] ist tot, alternative Quelle nötig!

        package_path = dingo.__path__[0]

        # decide whether cable or line is used (initially for entire grid) and set grid's attribute
        if self.v_level == 20:
            self.default_branch_kind = 'line'
        elif self.v_level == 10:
            self.default_branch_kind = 'cable'

        # get max. count of half rings per MV grid district
        mv_half_ring_count_max = int(cfg_dingo.get('mv_routing_tech_constraints',
                                                   'mv_half_ring_count_max'))
        #mv_half_ring_count_max=20

        # load cable/line assumptions, file_names and parameter
        if self.default_branch_kind == 'line':
            load_factor_normal = float(cfg_dingo.get('assumptions',
                                                     'load_factor_mv_line_lc_normal'))
            branch_parameters = self.network.static_data['MV_overhead_lines']

            # load cables as well to use it within settlements
            branch_parameters_settle = self.network.static_data['MV_cables']
            # select types with appropriate voltage level
            branch_parameters_settle = branch_parameters_settle[branch_parameters_settle['U_n'] == self.v_level]

        elif self.default_branch_kind == 'cable':
            load_factor_normal = float(cfg_dingo.get('assumptions',
                                                     'load_factor_mv_cable_lc_normal'))
            branch_parameters = self.network.static_data['MV_cables']
        else:
            raise ValueError('Grid\'s default_branch_kind is invalid, could not set branch parameters.')

        # select appropriate branch params according to voltage level, sorted ascending by max. current
        # use <240mm2 only (ca. 420A) for initial rings and for disambiguation of agg. LA
        branch_parameters = branch_parameters[branch_parameters['U_n'] == self.v_level]
        branch_parameters = branch_parameters[branch_parameters['I_max_th'] < 420].sort_values('I_max_th')

        # get largest line/cable type
        branch_type_max = branch_parameters.loc[branch_parameters['I_max_th'].idxmax()]

        # set aggregation flag using largest available line/cable
        self.set_nodes_aggregation_flag(branch_type_max['I_max_th'] * load_factor_normal)

        # calc peak current sum (= "virtual" current) of whole grid (I = S / sqrt(3) / U) excluding load areas of type
        # satellite and aggregated
        peak_current_sum = ((self.grid_district.peak_load -
                             self.grid_district.peak_load_satellites -
                             self.grid_district.peak_load_aggregated) /
                            (3**0.5) / self.v_level)  # units: kVA / kV = A

        branch_type_settle = branch_type_settle_max = None

        # search the smallest possible line/cable for MV grid district in equipment datasets for all load areas
        # excluding those of type satellite and aggregated
        for idx, row in branch_parameters.iterrows():
            # calc number of required rings using peak current sum of grid district,
            # load factor and max. current of line/cable
            half_ring_count = round(peak_current_sum / (row['I_max_th'] * load_factor_normal))

            if debug:
                logger.debug('=== Selection of default branch type in {} ==='.format(self))
                logger.debug('Peak load= {} kVA'.format(self.grid_district.peak_load))
                logger.debug('Peak current={}'.format(peak_current_sum))
                logger.debug('I_max_th={}'.format(row['I_max_th']))
                logger.debug('Half ring count={}'.format(half_ring_count))

            # if count of half rings is below or equal max. allowed count, use current branch type as default
            if half_ring_count <= mv_half_ring_count_max:
                if self.default_branch_kind == 'line':
                    # TODO: Newly installed cable has a greater I_max_th than former line, check with grid planning
                    # TODO: principles and add to documentation
                    # OLD:
                    # branch_type_settle = branch_parameters_settle.ix\
                    #                      [branch_parameters_settle\
                    #                      [branch_parameters_settle['I_max_th'] - row['I_max_th'] > 0].\
                    #                      sort_values(by='I_max_th')['I_max_th'].idxmin()]

                    # take only cables that can handle at least the current of the line
                    branch_parameters_settle_filter = branch_parameters_settle[\
                                                      branch_parameters_settle['I_max_th'] - row['I_max_th'] > 0]
                    # get cable type with similar (but greater) I_max_th
                    branch_type_settle = branch_parameters_settle_filter.loc[\
                                         branch_parameters_settle_filter['I_max_th'].idxmin()]

                return row, branch_type_max, branch_type_settle

        # no equipment was found, return largest available line/cable

        if debug:
            logger.debug('No appropriate line/cable type could be found for '
                         '{}, declare some load areas as aggregated.'.format(
                self))

        if self.default_branch_kind == 'line':
            branch_type_settle_max = branch_parameters_settle.loc[branch_parameters_settle['I_max_th'].idxmax()]

        return branch_type_max, branch_type_max, branch_type_settle_max

    def set_nodes_aggregation_flag(self, peak_current_branch_max):
        """ Set LV load areas with too high demand to aggregated type.

        Args:
            peak_current_branch_max: Max. allowed current for line/cable

        Returns:
            nothing
        """

        for lv_load_area in self.grid_district.lv_load_areas():
            peak_current_node = (lv_load_area.peak_load_sum / (3**0.5) / self.v_level)  # units: kVA / kV = A
            if peak_current_node > peak_current_branch_max:
                lv_load_area.is_aggregated = True

        # add peak demand for all LV load areas of aggregation type
        self.grid_district.add_aggregated_peak_demand()

    def export_to_pypsa(self, session, method='onthefly'):
        """Exports MVGridDingo grid to PyPSA database tables

        Peculiarities of MV grids are implemented here. Derive general export
        method from this and adapt to needs of LVGridDingo

        Parameters
        ----------
        session: SQLalchemy database session
        method: str
            Specify export method
            If method='db' grid data will be exported to database
            If method='onthefly' grid data will be passed to PyPSA directly (default)

        Notes
        -----
        It has to be proven that this method works for LV grids as well!

        Dingo treats two stationary case of powerflow:
        1) Full load: We assume no generation and loads to be set to peak load
        2) Generation worst case:
        """

        # definitions for temp_resolution table
        temp_id = 1
        timesteps = 2
        start_time = datetime(1970, 1, 1, 00, 00, 0)
        resolution = 'H'

        nodes = self._graph.nodes()

        edges = [edge for edge in list(self.graph_edges())
                 if (edge['adj_nodes'][0] in nodes and not isinstance(
                edge['adj_nodes'][0], LVLoadAreaCentreDingo))
                 and (edge['adj_nodes'][1] in nodes and not isinstance(
                edge['adj_nodes'][1], LVLoadAreaCentreDingo))]

        if method is 'db':

            # Export node objects: Busses, Loads, Generators
            pypsa_io.export_nodes(self,
                                  session,
                                  nodes,
                                  temp_id,
                                  lv_transformer=False)

            # Export edges
            pypsa_io.export_edges(self, session, edges)

            # Create table about temporal coverage of PF analysis
            pypsa_io.create_temp_resolution_table(session,
                                                  timesteps=timesteps,
                                                  resolution=resolution,
                                                  start_time=start_time)
        elif method is 'onthefly':

            nodes_dict, components_data = pypsa_io.nodes_to_dict_of_dataframes(
                self,
                nodes,
                lv_transformer=False)
            edges_dict = pypsa_io.edges_to_dict_of_dataframes(self, edges)
            components = tools.merge_two_dicts(nodes_dict, edges_dict)

            return components, components_data
        else:
            raise ValueError('Sorry, this export method does not exist!')

    def run_powerflow(self, session, export_pypsa_dir=None,  method='onthefly', debug=False):
        """ Performs power flow calculation for all MV grids

        Args:
            session: SQLalchemy database session
            export_pypsa_dir: str
                Sub-directory in output/debug/grid/ where csv Files of PyPSA network are exported to.
                Export is omitted if argument is empty.
            method: str
                Specify export method
                If method='db' grid data will be exported to database
                If method='onthefly' grid data will be passed to PyPSA directly (default)
            debug: If True, information is printed during process

        Notes:
            It has to be proven that this method works for LV grids as well!

            Dingo treats two stationary case of powerflow:
            1) Full load: We assume no generation and loads to be set to peak load
            2) Generation worst case:
        """

        if method is 'db':
            # export grid data to db (be ready for power flow analysis)
            self.export_to_pypsa(session, method=method)

            # run the power flow problem
            pypsa_io.run_powerflow(session, export_pypsa_dir=export_pypsa_dir)

            # import results from db
            self.import_powerflow_results(session)

        elif method is 'onthefly':
            components, components_data = self.export_to_pypsa(session, method)
            pypsa_io.run_powerflow_onthefly(components,
                                            components_data,
                                            self,
                                            export_pypsa_dir=export_pypsa_dir,
                                            debug=debug)

    def import_powerflow_results(self, session):
        """
        Assign results from power flow analysis to edges and nodes

        Parameters
        ----------
        session: SQLalchemy database session
        Returns
        -------
        None
        """

        # bus data
        pypsa_io.import_pfa_bus_results(session, self)

        # line data
        pypsa_io.import_pfa_line_results(session, self)

        # transformer data

    def reinforce_grid(self):
        """ Performs grid reinforcement measures for current MV grid
        Args:

        Returns:

        """
        # TODO: Finalize docstring

        reinforce_grid(self, mode='MV')

    def set_circuit_breakers(self, debug=False):
        """ Calculates the optimal position of the existing circuit breakers and relocates them within the graph,
            see method `set_circuit_breakers` in dingo.grid.mv_grid.tools for details.
        Args:
            debug: If True, information is printed during process
        """
        set_circuit_breakers(self, debug)

    def __repr__(self):
        return 'mv_grid_' + str(self.id_db)


class LVGridDingo(GridDingo):
    """ DINGO low voltage grid

    Parameters
    ----------
    region : LV region (instance of LVLoadAreaDingo class) that is associated with grid

    Notes:
      It is assumed that LV grid have got cables only (attribute 'default_branch_kind')
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.default_branch_kind = kwargs.get('default_branch_kind', 'cable')
        self._station = None
        self._loads = []
        self.population = kwargs.get('population', None)

    def station(self):
        """Returns grid's station"""
        return self._station

    def add_station(self, lv_station):
        """Adds a LV station to _station and grid graph if not already existing"""
        if not isinstance(lv_station, LVStationDingo):
            raise Exception('Given LV station is not a LVStationDingo object.')
        if self._station is None:
            self._station = lv_station
            self.graph_add_node(lv_station)
            self.grid_district.lv_load_area.mv_grid_district.mv_grid.graph_add_node(lv_station)

    def add_load(self, lv_load):
        """Adds a LV load to _loads and grid graph if not already existing"""
        if lv_load not in self._loads and isinstance(lv_load,
                                                     LVLoadDingo):
            self._loads.append(lv_load)
            self.graph_add_node(lv_load)

    def add_cable_dist(self, lv_cable_dist):
        """Adds a LV cable_dist to _cable_dists and grid graph if not already existing"""
        if lv_cable_dist not in self._cable_distributors and isinstance(lv_cable_dist,
                                                                        LVCableDistributorDingo):
            self._cable_distributors.append(lv_cable_dist)
            self.graph_add_node(lv_cable_dist)

    def loads(self):
        """Returns a generator for iterating over LV _load"""
        for load in self._loads:
            yield load

    def select_transformer(self, peak_load):
        """
        Selects LV transformer according to peak load of LV grid district.

        Parameters
        ----------
        peak_load: int
            Peak load of LV grid district

        Returns
        -------
        transformer: DataFrame
            Parameters of chosen Transformer

        Notes
        -----
        The LV transformer with the next higher available nominal apparent power is chosen.
        Therefore, a max. allowed transformer loading of 100% is implicitly assumed.
        """

        # get equipment parameters of LV transformers
        trafo_parameters = self.network.static_data['LV_trafos']

        # choose trafo
        transformer = trafo_parameters.iloc[
            trafo_parameters[
                trafo_parameters['S_max'] > peak_load]['S_max'].idxmin()]

        return transformer

    def select_typified_grid_model(self,
                                   string_properties,
                                   apartment_string,
                                   trafo_parameters,
                                   population):
        """
        Selects typified model grid based on population

        Parameters
        ----------
        string_properties: DataFrame
            Properties of LV typified model grids
        apartment_string: DataFrame
            Relational table of apartment count and strings of model grid
        trafo_parameters: DataFrame
            Equipment parameters of LV transformers
        population: Int
            Population within LV grid district

        Notes
        -----
        In total 196 distinct LV grid topologies are available that are chosen
        by population in the LV grid district. Population is translated to
        number of house branches. Each grid model fits a number of house
        branches. If this number exceed 196, still the grid topology of 196
        house branches is used. The peak load of the LV grid district is
        uniformly distributed across house branches.

        Returns
        -------
        selected_strings_df: DataFrame
            Selected string of typified model grid
        transformer: Dataframe
            Parameters of chosen Transformer
        """

        apartment_house_branch_ratio = cfg_dingo.get("assumptions",
            "apartment_house_branch_ratio")
        population_per_apartment = cfg_dingo.get("assumptions",
            "population_per_apartment")

        house_branches = round(population / (population_per_apartment *
                                         apartment_house_branch_ratio))

        if house_branches <= 0:
            house_branches = 1

        if house_branches > 196:
            house_branches = 196

        # select set of strings that represent one type of model grid
        strings = apartment_string.loc[house_branches]
        selected_strings = [int(s) for s in strings[strings >= 1].index.tolist()]

        # slice dataframe of string parameters
        selected_strings_df = string_properties.loc[selected_strings]

        # add number of occurences of each branch to df
        occurence_selector = [str(i) for i in selected_strings]
        selected_strings_df['occurence'] = strings.loc[occurence_selector].tolist()

        # TODO: WORKAROUND! Use peak load of LVGD to determine trafo size
        transformer_temp = 630
        transformer = {}

        transformer['S_max'] = trafo_parameters.loc[
            transformer_temp].name
        transformer['x'] = trafo_parameters.loc[
            transformer_temp, 'X']
        transformer['r'] = trafo_parameters.loc[
            transformer_temp, 'R']

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
                        grid=self,
                        string_id=i,
                        branch_no=branch_no,
                        load_no=house_branch)

                    lv_load = LVLoadDingo(grid=self,
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
                            self.station(),
                            lv_cable_dist,
                            branch=BranchDingo(
                                length=row['distance house branch'],
                                type=cable_name
                                ))
                    else:
                        self._graph.add_edge(
                            self._cable_distributors[-2],
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
                            type=self.network.static_data['LV_cables']. \
                                loc[house_cable_name]))

    def reinforce_grid(self):
        """ Performs grid reinforcement measures for current LV grid
        Args:

        Returns:

        """
        # TODO: Finalize docstring

        reinforce_grid(self, mode='LV')

    def __repr__(self):
        return 'lv_grid_' + str(self.id_db)
