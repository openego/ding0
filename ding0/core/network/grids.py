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


# from ding0.core.network import GridDing0
from . import GridDing0
from ding0.core.network.stations import *
from ding0.core.network import RingDing0, CircuitBreakerDing0
from ding0.core.network.loads import *
from ding0.core.network.cable_distributors import MVCableDistributorDing0, LVCableDistributorDing0
from ding0.grid.mv_grid import mv_routing, mv_connect
from ding0.grid.lv_grid import build_grid, lv_connect
from ding0.tools import config as cfg_ding0, pypsa_io, tools
from ding0.tools.geo import calc_geo_dist_vincenty
from ding0.grid.mv_grid.tools import set_circuit_breakers
from ding0.flexopt.reinforce_grid import *
from ding0.core.structure.regions import LVLoadAreaCentreDing0

import os
import networkx as nx
from datetime import datetime
import pyproj
from functools import partial
import logging

if not 'READTHEDOCS' in os.environ:
    from shapely.ops import transform

logger = logging.getLogger('ding0')


class MVGridDing0(GridDing0):
    """ DING0 medium voltage grid

    Parameters
    ----------
    region : :obj:`MVGridDistrictDing0`
        MV region (instance of MVGridDistrictDing0 class) that is associated with grid
    default_branch_kind: :obj:`str`
        kind of branch (possible values: 'cable' or 'line')
    default_branch_type: :pandas:`pandas.Series<series>`   
        type of branch (pandas Series object with cable/line parameters)
    
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
        """Creates circuit breaker object and ...

        Args
        ----
        circ_breaker: CircuitBreakerDing0
            Description #TODO
        """
        if circ_breaker not in self._circuit_breakers and isinstance(circ_breaker, CircuitBreakerDing0):
            self._circuit_breakers.append(circ_breaker)
            self.graph_add_node(circ_breaker)

    def open_circuit_breakers(self):
        """Opens all circuit breakers in MV grid """
        for circ_breaker in self.circuit_breakers():
            circ_breaker.open()

    def close_circuit_breakers(self):
        """Closes all circuit breakers in MV grid """
        for circ_breaker in self.circuit_breakers():
            circ_breaker.close()

    def add_station(self, mv_station, force=False):
        """Adds MV station if not already existing

        Args
        ----
        mv_station: MVStationDing0
            Description #TODO
        force: bool
            If True, MV Station is set even though it's not empty (override)
        """
        if not isinstance(mv_station, MVStationDing0):
            raise Exception('Given MV station is not a MVStationDing0 object.')
        if self._station is None:
            self._station = mv_station
            self.graph_add_node(mv_station)
        else:
            if force:
                self._station = mv_station
            else:
                raise Exception('MV Station already set, use argument `force=True` to override.')

    def add_load(self, lv_load):
        """Adds a MV load to _loads and grid graph if not already existing
        
        Args
        ----
        lv_load : float
            Desription #TODO
        """
        if lv_load not in self._loads and isinstance(lv_load,
                                                     MVLoadDing0):
            self._loads.append(lv_load)
            self.graph_add_node(lv_load)

    def add_cable_distributor(self, cable_dist):
        """Adds a cable distributor to _cable_distributors if not already existing
        
        Args
        ----
        cable_dist : float
            Desription #TODO
        """
        if cable_dist not in self.cable_distributors() and isinstance(cable_dist,
                                                                      MVCableDistributorDing0):
            # add to array and graph
            self._cable_distributors.append(cable_dist)
            self.graph_add_node(cable_dist)

    def remove_cable_distributor(self, cable_dist):
        """Removes a cable distributor from _cable_distributors if existing"""
        if cable_dist in self.cable_distributors() and isinstance(cable_dist,
                                                                  MVCableDistributorDing0):
            # remove from array and graph
            self._cable_distributors.remove(cable_dist)
            if self._graph.has_node(cable_dist):
                self._graph.remove_node(cable_dist)

    def add_ring(self, ring):
        """Adds a ring to _rings if not already existing"""
        if ring not in self._rings and isinstance(ring, RingDing0):
            self._rings.append(ring)

    def rings_count(self):
        """Returns the count of rings in MV grid
        
        Returns
        -------
        int
            Count of ringos in MV grid.
        """
        return len(self._rings)

    def rings_nodes(self, include_root_node=False, include_satellites=False):
        """ Returns a generator for iterating over rings (=routes of MVGrid's graph)

        Args
        ----
        include_root_node: bool, defaults to False
            If True, the root node is included in the list of ring nodes.
        include_satellites: bool, defaults to False
            If True, the satellite nodes (nodes that diverge from ring nodes) is included in the list of ring nodes.
            
        Yields
        ------
        :obj:`list` of :obj:`GridDing0`
            List with nodes of each ring of _graph in- or excluding root node (HV/MV station) (arg `include_root_node`),
            format::
             
            [ ring_m_node_1, ..., ring_m_node_n ]
            
        Notes
        -----
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

    def rings_full_data(self):
        """ Returns a generator for iterating over each ring

        Yields
        ------
            For each ring, tuple composed by ring ID, list of edges, list of nodes
        Notes
        -----
            Circuit breakers must be closed to find rings, this is done automatically.
        """
        #close circuit breakers
        for circ_breaker in self.circuit_breakers():
            if not circ_breaker.status == 'closed':
                circ_breaker.close()
                logger.info('Circuit breakers were closed in order to find MV '
                            'rings')
        #find True rings (cycles from station through breaker and back to station)
        for ring_nodes in nx.cycle_basis(self._graph, root=self._station):
            edges_ring = []
            for node in ring_nodes:
                for edge in self.graph_branches_from_node(node):
                    nodes_in_the_branch = self.graph_nodes_from_branch(edge[1]['branch'])
                    if (nodes_in_the_branch[0] in ring_nodes and
                        nodes_in_the_branch[1] in ring_nodes
                        ):
                        if not edge[1]['branch'] in edges_ring:
                            edges_ring.append(edge[1]['branch'])
            yield (edges_ring[0].ring,edges_ring,ring_nodes)

        ##Find "rings" associated to aggregated LA
        #for node in self.graph_nodes_sorted():
        #    if isinstance(node,LVLoadAreaCentreDing0): # MVCableDistributorDing0
        #        edges_ring = []
        #        ring_nodes = []
        #        if node.lv_load_area.is_aggregated:
        #            ring_info = self.find_path(self._station, node, type='edges')
        #            for info in ring_info:
        #                edges_ring.append(info[2]['branch'])
        #                ring_nodes.append(info[0])
        #            ring_nodes.append(ring_info[-1][1])
        #            yield (edges_ring[0].ring,edges_ring,ring_nodes)

    def get_ring_from_node(self, node):
        """ Determines the ring (RingDing0 object) which node is member of.
        Args
        ----
        node: GridDing0
            Ding0 object (member of graph)
        
        Returns
        -------
        RingDing0
            Ringo of which node is member.
        """
        try:
            return self.graph_branches_from_node(node)[0][1]['branch'].ring
        except:
            raise Exception('Cannot get node\'s associated ring.')

    def graph_nodes_from_subtree(self, node_source):
        """ Finds all nodes of a tree that is connected to `node_source` and are (except `node_source`) not part of the 
        ring of `node_source` (traversal of graph from `node_source` excluding nodes along ring).
            
        Example
        -------
        A given graph with ring (edges) 0-1-2-3-4-5-0 and a tree starting at node (`node_source`) 3 with edges 3-6-7, 3-6-8-9 will return [6,7,8,9]

        Args
        ----
        node_source: GridDing0
            source node (Ding0 object), member of _graph

        Returns
        -------
        :obj:`list` of :obj:`GridDing0`
            List of nodes (Ding0 objects)
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
        """ Generates and sets ids of branches for MV and underlying LV grids.
        
        While IDs of imported objects can be derived from dataset's ID, branches
        are created within DING0 and need unique IDs (e.g. for PF calculation).
        """

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
        """ Performs routing on Load Area centres to build MV grid with ring topology.

        Args
        ----
        debug: bool, defaults to False
            If True, information is printed while routing
        anim: type, defaults to None
            Descr #TODO
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

        Args
        ----
        debug: bool, defaults to False
            If True, information is printed during process
        """

        self._graph = mv_connect.mv_connect_generators(self.grid_district, self._graph, debug)

    def parametrize_grid(self, debug=False):
        """ Performs Parametrization of grid equipment:
        
            i) Sets voltage level of MV grid, 
            ii) Operation voltage level and transformer of HV/MV station, 
            iii) Default branch types (normal, aggregated, settlement)

        Args
        ----
        debug: bool, defaults to False
            If True, information is printed during process.
            
        Notes
        -----
        It is assumed that only cables are used within settlements.
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
        self._station.select_transformers()

    def set_voltage_level(self, mode='distance'):
        """ Sets voltage level of MV grid according to load density of MV Grid District or max.
        distance between station and Load Area.

        Parameters
        ----------
        mode: :obj:`str`
            method to determine voltage level
            
            * 'load_density': Decision on voltage level is determined by load density
              of the considered region. Urban areas (load density of
              >= 1 MW/km2 according to [#]_) usually got a voltage of
              10 kV whereas rural areas mostly use 20 kV.

            * 'distance' (default): Decision on voltage level is determined by the max.
              distance between Grid District's HV-MV station and Load
              Areas (LA's centre is used). According to [#]_ a value of
              1kV/kV can be assumed. The `voltage_per_km_threshold`
              defines the distance threshold for distinction.
              (default in config = (20km+10km)/2 = 15km)

        References
        ----------
        .. [#] Falk Schaller et al., "Modellierung realitätsnaher zukünftiger Referenznetze im Verteilnetzsektor zur
            Überprüfung der Elektroenergiequalität", Internationaler ETG-Kongress Würzburg, 2011
        .. [#] Klaus Heuck et al., "Elektrische Energieversorgung", Vieweg+Teubner, Wiesbaden, 2007

        """

        if mode == 'load_density':

            # get power factor for loads
            cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')

            # get load density
            load_density_threshold = float(cfg_ding0.get('assumptions',
                                                         'load_density_threshold'))

            # transform MVGD's area to epsg 3035
            # to achieve correct area calculation
            projection = partial(
                pyproj.transform,
                pyproj.Proj(init='epsg:4326'),  # source coordinate system
                pyproj.Proj(init='epsg:3035'))  # destination coordinate system

            # calculate load density
            kw2mw = 1e-3
            sqm2sqkm = 1e6
            load_density = ((self.grid_district.peak_load * kw2mw / cos_phi_load) /
                            (transform(projection, self.grid_district.geo_data).area / sqm2sqkm)) # unit MVA/km^2

            # identify voltage level
            if load_density < load_density_threshold:
                self.v_level = 20
            elif load_density >= load_density_threshold:
                self.v_level = 10
            else:
                raise ValueError('load_density is invalid!')

        elif mode == 'distance':

            # get threshold for 20/10kV disambiguation
            voltage_per_km_threshold = float(cfg_ding0.get('assumptions',
                                                           'voltage_per_km_threshold'))

            # initial distance
            dist_max = 0
            import time
            start = time.time()
            for node in self.graph_nodes_sorted():
                if isinstance(node, LVLoadAreaCentreDing0):
                    # calc distance from MV-LV station to LA centre
                    dist_node = calc_geo_dist_vincenty(self.station(), node) / 1e3
                    if dist_node > dist_max:
                        dist_max = dist_node

            # max. occurring distance to a Load Area exceeds threshold => grid operates at 20kV
            if dist_max >= voltage_per_km_threshold:
                self.v_level = 20
            # not: grid operates at 10kV
            else:
                self.v_level = 10

        else:
            raise ValueError('parameter \'mode\' is invalid!')

    def set_default_branch_type(self, debug=False):
        """ Determines default branch type according to grid district's peak load and standard equipment.

        Args
        ----
        debug: bool, defaults to False
            If True, information is printed during process

        Returns
        -------
        :pandas:`pandas.Series<series>`   
            default branch type: pandas Series object. If no appropriate type is found, return largest possible one.
        :pandas:`pandas.Series<series>`    
            default branch type max: pandas Series object. Largest available line/cable type

        Notes
        -----
        Parameter values for cables and lines are taken from [#]_, [#]_ and [#]_.

        Lines are chosen to have 60 % load relative to their nominal capacity according to [#]_.

        Decision on usage of overhead lines vs. cables is determined by load density of the considered region. Urban
        areas usually are equipped with underground cables whereas rural areas often have overhead lines as MV
        distribution system [#]_.

        References
        ----------
        .. [#] Klaus Heuck et al., "Elektrische Energieversorgung", Vieweg+Teubner, Wiesbaden, 2007
        .. [#] René Flosdorff et al., "Elektrische Energieverteilung", Vieweg+Teubner, 2005
        .. [#] Südkabel GmbH, "Einadrige VPE-isolierte Mittelspannungskabel",
            http://www.suedkabel.de/cms/upload/pdf/Garnituren/Einadrige_VPE-isolierte_Mittelspannungskabel.pdf, 2017
        .. [#] Deutsche Energie-Agentur GmbH (dena), "dena-Verteilnetzstudie. Ausbau- und Innovationsbedarf der
            Stromverteilnetze in Deutschland bis 2030.", 2012
        .. [#] Tao, X., "Automatisierte Grundsatzplanung von
            Mittelspannungsnetzen", Dissertation, RWTH Aachen, 2007
        """

        # decide whether cable or line is used (initially for entire grid) and set grid's attribute
        if self.v_level == 20:
            self.default_branch_kind = 'line'
        elif self.v_level == 10:
            self.default_branch_kind = 'cable'

        # get power factor for loads
        cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')

        # get max. count of half rings per MV grid district
        mv_half_ring_count_max = int(cfg_ding0.get('mv_routing_tech_constraints',
                                                   'mv_half_ring_count_max'))
        #mv_half_ring_count_max=20

        # load cable/line assumptions, file_names and parameter
        if self.default_branch_kind == 'line':
            load_factor_normal = float(cfg_ding0.get('assumptions',
                                                     'load_factor_mv_line_lc_normal'))
            branch_parameters = self.network.static_data['MV_overhead_lines']

            # load cables as well to use it within settlements
            branch_parameters_settle = self.network.static_data['MV_cables']
            # select types with appropriate voltage level
            branch_parameters_settle = branch_parameters_settle[branch_parameters_settle['U_n'] == self.v_level]

        elif self.default_branch_kind == 'cable':
            load_factor_normal = float(cfg_ding0.get('assumptions',
                                                     'load_factor_mv_cable_lc_normal'))
            branch_parameters = self.network.static_data['MV_cables']
        else:
            raise ValueError('Grid\'s default_branch_kind is invalid, could not set branch parameters.')

        # select appropriate branch params according to voltage level, sorted ascending by max. current
        # use <240mm2 only (ca. 420A) for initial rings and for disambiguation of agg. LA
        branch_parameters = branch_parameters[branch_parameters['U_n'] == self.v_level]
        branch_parameters = branch_parameters[branch_parameters['reinforce_only'] == 0].sort_values('I_max_th')

        # get largest line/cable type
        branch_type_max = branch_parameters.loc[branch_parameters['I_max_th'].idxmax()]

        # set aggregation flag using largest available line/cable
        self.set_nodes_aggregation_flag(branch_type_max['I_max_th'] * load_factor_normal)

        # calc peak current sum (= "virtual" current) of whole grid (I = S / sqrt(3) / U) excluding load areas of type
        # satellite and aggregated
        peak_current_sum = ((self.grid_district.peak_load -
                             self.grid_district.peak_load_satellites -
                             self.grid_district.peak_load_aggregated) /
                            cos_phi_load /
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

                    # take only cables that can handle at least the current of the line
                    branch_parameters_settle_filter = branch_parameters_settle[\
                                                      branch_parameters_settle['I_max_th'] - row['I_max_th'] > 0]

                    # get cable type with similar (but greater) I_max_th
                    # note: only grids with lines as default branch kind get cables in settlements
                    # (not required in grids with cables as default branch kind)
                    branch_type_settle = branch_parameters_settle_filter.loc[\
                                         branch_parameters_settle_filter['I_max_th'].idxmin()]

                return row, branch_type_max, branch_type_settle

        # no equipment was found, return largest available line/cable

        if debug:
            logger.debug('No appropriate line/cable type could be found for '
                         '{}, declare some load areas as aggregated.'.format(self))

        if self.default_branch_kind == 'line':
            branch_type_settle_max = branch_parameters_settle.loc[branch_parameters_settle['I_max_th'].idxmax()]

        return branch_type_max, branch_type_max, branch_type_settle_max

    def set_nodes_aggregation_flag(self, peak_current_branch_max):
        """ Set Load Areas with too high demand to aggregated type.

        Parameters
        ----------
        peak_current_branch_max: :obj:`float`
            Max. allowed current for line/cable

        """

        for lv_load_area in self.grid_district.lv_load_areas():
            peak_current_node = (lv_load_area.peak_load / (3**0.5) / self.v_level)  # units: kVA / kV = A
            if peak_current_node > peak_current_branch_max:
                lv_load_area.is_aggregated = True

        # add peak demand for all Load Areas of aggregation type
        self.grid_district.add_aggregated_peak_demand()

    def export_to_pypsa(self, session, method='onthefly'):
        """Exports MVGridDing0 grid to PyPSA database tables

        Peculiarities of MV grids are implemented here. Derive general export
        method from this and adapt to needs of LVGridDing0

        Parameters
        ----------
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        method: :obj:`str`
            Specify export method::
            
            * 'db': grid data will be exported to database
            * 'onthefly': grid data will be passed to PyPSA directly (default)

        Notes
        -----
        It has to be proven that this method works for LV grids as well!

        Ding0 treats two stationary case of powerflow:

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
                edge['adj_nodes'][0], LVLoadAreaCentreDing0))
                 and (edge['adj_nodes'][1] in nodes and not isinstance(
                edge['adj_nodes'][1], LVLoadAreaCentreDing0))]

        if method == 'db':

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
        elif method == 'onthefly':

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

        Args
        ----
        session : :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Database session
        export_pypsa_dir: :obj:`str`
            Sub-directory in output/debug/grid/ where csv Files of PyPSA network are exported to.
            
            Export is omitted if argument is empty.
        method: :obj:`str`
            Specify export method::
            
            'db': grid data will be exported to database
            'onthefly': grid data will be passed to PyPSA directly (default)
            
        debug: bool, defaults to False
            If True, information is printed during process

        Notes
        -----
        It has to be proven that this method works for LV grids as well!

            Ding0 treats two stationary case of powerflow:
            1) Full load: We assume no generation and loads to be set to peak load
            2) Generation worst case:
        """

        if method == 'db':
            raise NotImplementedError("Please use 'onthefly'.")

        elif method == 'onthefly':
            components, components_data = self.export_to_pypsa(session, method)
            pypsa_io.run_powerflow_onthefly(components,
                                            components_data,
                                            self,
                                            export_pypsa_dir=export_pypsa_dir,
                                            debug=debug)

    def import_powerflow_results(self, session):
        """Assign results from power flow analysis to edges and nodes

        Parameters
        ----------
        session: :sqlalchemy:`SQLAlchemy session object<orm/session_basics.html>`
            Description
        
        """

        # bus data
        pypsa_io.import_pfa_bus_results(session, self)

        # line data
        pypsa_io.import_pfa_line_results(session, self)

        # transformer data

    def reinforce_grid(self):
        """Performs grid reinforcement measures for current MV grid

        """
        # TODO: Finalize docstring

        reinforce_grid(self, mode='MV')

    def set_circuit_breakers(self, debug=False):
        """ Calculates the optimal position of the existing circuit breakers and relocates them within the graph.
        
        Args
        ----
        debug: bool, defaults to False
            If True, information is printed during process
            
        See Also
        --------
        ding0.grid.mv_grid.tools.set_circuit_breakers
        """
        set_circuit_breakers(self, debug=debug)

    def __repr__(self):
        return 'mv_grid_' + str(self.id_db)


class LVGridDing0(GridDing0):
    """ DING0 low voltage grid

    Parameters
    ----------
    region : LVLoadAreaDing0
        LV region that is associated with grid
    default_branch_kind : :obj:`str`
        description #TODO
    population : 
        description #TODO

    Notes
    -----
        It is assumed that LV grid have got cables only (attribute 'default_branch_kind')
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.default_branch_kind = kwargs.get('default_branch_kind', 'cable')
        self._station = None
        self.population = kwargs.get('population', None)

    def station(self):
        """Returns grid's station"""
        return self._station

    def add_station(self, lv_station):
        """Adds a LV station to _station and grid graph if not already existing"""
        if not isinstance(lv_station, LVStationDing0):
            raise Exception('Given LV station is not a LVStationDing0 object.')
        if self._station is None:
            self._station = lv_station
            self.graph_add_node(lv_station)
            self.grid_district.lv_load_area.mv_grid_district.mv_grid.graph_add_node(lv_station)

    def loads_sector(self, sector='res'):
        """Returns a generator for iterating over grid's sectoral loads
        
        Parameters
        ----------
        sector: :obj:`str`ing
            possible values::
                
                'res' (residential),
                'ria' (retail, industrial, agricultural)

        Yields
        -------
        int 
            Generator for iterating over loads of the type specified in `sector`. 
        """
        
        for load in self._loads:
            if (sector == 'res') and (load.string_id is not None):
                yield load
            elif (sector == 'ria') and (load.string_id is None):
                yield load

    def add_load(self, lv_load):
        """Adds a LV load to _loads and grid graph if not already existing
        
        Parameters
        ----------
        lv_load : 
            Description #TODO
        """
        if lv_load not in self._loads and isinstance(lv_load,
                                                     LVLoadDing0):
            self._loads.append(lv_load)
            self.graph_add_node(lv_load)

    def add_cable_dist(self, lv_cable_dist):
        """Adds a LV cable_dist to _cable_dists and grid graph if not already existing
        
        Parameters
        ----------
        lv_cable_dist : 
            Description #TODO
        """
        if lv_cable_dist not in self._cable_distributors and isinstance(lv_cable_dist,
                                                                        LVCableDistributorDing0):
            self._cable_distributors.append(lv_cable_dist)
            self.graph_add_node(lv_cable_dist)

    def build_grid(self):
        """Create LV grid graph
        """

        # add required transformers
        build_grid.transformer(self)

        # add branches of sectors retail/industrial and agricultural
        build_grid.build_ret_ind_agr_branches(self.grid_district)

        # add branches of sector residential
        build_grid.build_residential_branches(self.grid_district)

        #self.graph_draw(mode='LV')

    def connect_generators(self, debug=False):
        """ Connects LV generators (graph nodes) to grid (graph)

        Args
        ----
        debug: bool, defaults to False
             If True, information is printed during process
        """

        self._graph = lv_connect.lv_connect_generators(self.grid_district, self._graph, debug)

    def reinforce_grid(self):
        """ Performs grid reinforcement measures for current LV grid.
        """
        # TODO: Finalize docstring

        reinforce_grid(self, mode='LV')

    def __repr__(self):
        return 'lv_grid_' + str(self.id_db)
