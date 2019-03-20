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


import matplotlib.pyplot as plt
import networkx as nx

from ding0.core.structure.regions import LVLoadAreaDing0, LVLoadAreaCentreDing0


class GridDing0:
    """
    The fundamental abstract class used to encapsulated
    the networkx graph and the relevant attributes of a
    power grid irrespective of voltage level. By design,
    this class is not expected to be instantiated directly.
    This class was designed to be inherited by
    :class:`~.ding0.core.network.grids.MVGridDing0` or by
    :class:`~.ding0.core.network.grids.LVGridDing0`.

    Parameters
    ----------
    network : :obj:`str`
        Name of the grid
    id_db : :obj:`str`
        id according to database table
    grid_district : :shapely:`Shapely Polygon object<polygons>`
        class, area that is covered by the lv grid
    v_level: :obj:`int`
        The integer value of the voltage level of the Grid in kV.
        Typically either 10 or 20.
    _cable_distributors: :obj:`list` of
        :class:`~.ding0.core.network.CableDistributorDing0` Objects
    _loads : :obj:`list` of
        :class:`~.ding0.core.network.LoadDing0` Objects.
        These are objects meant to be considered as MV-Level loads
    _generators : :obj:`list` of
        :class:`~.ding0.core.network.GeneratorDing0` or
        :class:`~.ding0.core.network.GeneratorFluctuatingDing0`
        Objects.
        These are objects meant to be considered as MV-Level Generators.
    _graph : :networkx:`networkx.Graph`
        The networkx graph of the network. Initially this is an empty graph
        which gets populated differently depending upon
        which child class inherits this class, either MVGridDing0 or
        LVGridDing0.
        
    """

    def __init__(self, **kwargs):
        self.network = kwargs.get('network', None)
        self.id_db = kwargs.get('id_db', None)
        self.grid_district = kwargs.get('grid_district', None)
        self._cable_distributors = []
        self._loads = []
        self._generators = []
        self.v_level = kwargs.get('v_level', None)

        self._graph = nx.Graph()

    def cable_distributors(self):
        """
        Provides access to the cable distributors in the grid.
        
        Returns
        -------
        obj:`list` generator of
            :class:`~.ding0.core.network.CableDistributorDing0` objects
        """
        for cable_dist in self._cable_distributors:
            yield cable_dist

    def cable_distributors_count(self):
        """
        Returns the count of cable distributors in grid
        
        Returns
        -------
        :obj:`int`
            Count of the
            :class:`~.ding0.core.network.CableDistributorDing0` objects

        """
        return len(self._cable_distributors)

    def loads(self):
        """
        Returns a generator for iterating over grid's loads
        
        Returns
        -------
        obj:`list` generator of
            :class:`~.ding0.core.network.LoadDing0` objects
        """
        for load in self._loads:
            yield load

    def loads_count(self):
        """
        Returns the count of loads in grid
        
        Returns
        -------
        :obj:`int`
            Count of the
            :class:`~.ding0.core.network.LoadDing0` objects
        """
        return len(self._loads)

    def generators(self):
        """
        Returns a generator for iterating over grid's generators
        
        Returns
        -------
        obj:`list` generator of
            :class:`~.ding0.core.network.GeneratorDing0` and
            :class:`~.ding0.core.network.GeneratorFluctuatingDing0`
            objects
        """
        for generator in self._generators:
            yield generator

    def add_generator(self, generator):
        """
        Adds a generator to :attr:`_generators` and grid graph if not already existing
        
        Parameters
        ----------
        generator : :class:`~.ding0.core.network.GeneratorDing0` or
            :class:`~.ding0.core.network.GeneratorFluctuatingDing0`

        
        """
        if generator not in self._generators and isinstance(generator,
                                                            GeneratorDing0):
            self._generators.append(generator)
            self.graph_add_node(generator)

    def graph_add_node(self, node_object):
        """
        Adds a station or cable distributor object
        to grid graph if not already existing
        
        Parameters
        ----------
        node_object : :class:`~.ding0.core.network.GeneratorDing0` or
            :class:`~.ding0.core.network.GeneratorFluctuatingDing0` or
            :class:`~.ding0.core.network.LoadDing0` or
            :class:`~.ding0.core.network.StationDing0` or
            :class:`~.ding0.core.network.CircuitBreakerDing0` or
            :class:`~.ding0.core.network.CableDistributorDing0`
            Description #TODO
        
        """
        if ((node_object not in self._graph.nodes()) and
            (isinstance(node_object, (StationDing0,
                                      CableDistributorDing0,
                                      LVLoadAreaCentreDing0,
                                      CircuitBreakerDing0,
                                      GeneratorDing0)))):
            self._graph.add_node(node_object)

    def graph_draw(self, mode):
        """ Draws grid graph using networkx

        This method is for debugging purposes only.
        Use ding0.tools.plots.plot_mv_topology() for advanced plotting.

        Parameters
        ----------
        mode : str
            Mode selection 'MV' or 'LV'.
            
        Notes
        -----
        The geo coords (for used crs see database import in class `NetworkDing0`)
        are used as positions for drawing but networkx uses cartesian crs.
        Since no coordinate transformation is performed, the drawn graph representation is falsified!
        """

        g = self._graph

        if mode == 'MV':
            # get draw params from nodes and edges (coordinates, colors, demands, etc.)
            nodes_pos = {}; demands = {}; demands_pos = {}
            nodes_color = []
            for node in g.nodes():
                if isinstance(node, (StationDing0,
                                     LVLoadAreaCentreDing0,
                                     CableDistributorDing0,
                                     GeneratorDing0,
                                     CircuitBreakerDing0)):
                    nodes_pos[node] = (node.geo_data.x, node.geo_data.y)
                    # TODO: MOVE draw/color settings to config
                if node == self.station():
                    nodes_color.append((1, 0.5, 0.5))
                else:
                    #demands[node] = 'd=' + '{:.3f}'.format(node.grid.region.peak_load_sum)
                    #demands_pos[node] = tuple([a+b for a, b in zip(nodes_pos[node], [0.003]*len(nodes_pos[node]))])
                    nodes_color.append((0.5, 0.5, 1))

            edges_color = []
            for edge in self.graph_edges():
                if edge['branch'].critical:
                    edges_color.append((1, 0, 0))
                else:
                    edges_color.append((0, 0, 0))

            plt.figure()
            nx.draw_networkx(g, nodes_pos, node_color=nodes_color, edge_color=edges_color, font_size=8)
            #nx.draw_networkx_labels(g, demands_pos, labels=demands, font_size=8)
            plt.show()

        elif mode == 'LV':
            nodes_pos = {}
            nodes_color = []

            for node in g.nodes():
                # get neighbors of station (=first node of each branch)
                station_neighbors = sorted(g.neighbors(self.station()), key=lambda _: repr(_))

                # set x-offset according to count of branches
                if len(station_neighbors) % 2 == 0:
                    x_pos_start = -(len(station_neighbors) // 2 - 0.5)
                else:
                    x_pos_start = -(len(station_neighbors) // 2)

                # set positions
                if isinstance(node, CableDistributorDing0):
                    if node.in_building:
                        nodes_pos[node] = (x_pos_start + node.branch_no - 1 + 0.25, -node.load_no - 2)
                        nodes_color.append((0.5, 0.5, 0.5))
                    else:
                        nodes_pos[node] = (x_pos_start + node.branch_no - 1, -node.load_no - 2)
                        nodes_color.append((0.5, 0.5, 0.5))

                elif isinstance(node, LoadDing0):
                    nodes_pos[node] = (x_pos_start + node.branch_no - 1 + 0.5, -node.load_no - 2 - 0.25)
                    nodes_color.append((0.5, 0.5, 1))
                elif isinstance(node, GeneratorDing0):
                    # get neighbor of geno
                    neighbor = g.neighbors(node)[0]

                    # neighbor is cable distributor of building
                    if isinstance(neighbor, CableDistributorDing0):
                        nodes_pos[node] = (x_pos_start + neighbor.branch_no - 1 + 0.5, -neighbor.load_no - 2 + 0.25)
                    else:
                        nodes_pos[node] = (1,1)

                    nodes_color.append((0.5, 1, 0.5))
                elif isinstance(node, StationDing0):
                    nodes_pos[node] = (0, 0)
                    nodes_color.append((1, 0.5, 0.5))

            plt.figure()
            nx.draw_networkx(g, nodes_pos, node_color=nodes_color, font_size=8, node_size=100)
            plt.show()

    def graph_nodes_sorted(self):
        """ Returns an (ascending) sorted list of graph's nodes (name is used as key).
        
        Returns
        -------
        :any:`list`
            Description #TODO check

        """
        return sorted(self._graph.nodes(), key=lambda _: repr(_))

    def graph_nodes_from_branch(self, branch):
        """ Returns nodes that are connected by `branch`

        Args
        ----
        branch: BranchDing0
            Description #TODO
                
        Returns
        -------
        (:obj:`GridDing0`, :obj:`GridDing0`)
            2-tuple of nodes (Ding0 objects) #TODO:Check
        """
        edges = nx.get_edge_attributes(self._graph, 'branch')
        nodes = list(edges.keys())[list(edges.values()).index(branch)]
        return nodes

    def graph_branches_from_node(self, node):
        """ Returns branches that are connected to `node`

        Args
        ----
        node: GridDing0
            Ding0 object (member of graph)
        
        Returns
        -------
        :any:`list`
            List of tuples (node in :obj:`GridDing0`, branch in :obj:`BranchDing0`) ::
            
                (node , branch_0 ),
                ...,
                (node , branch_N ),
                
        """
        # TODO: This method can be replaced and speed up by using NetworkX' neighbors()

        branches = []
        branches_dict = self._graph.edge[node]
        for branch in branches_dict.items():
            branches.append(branch)
        return sorted(branches, key=lambda _: repr(_))

    def graph_edges(self):
        """ Returns a generator for iterating over graph edges

        The edge of a graph is described by the two adjacent node and the branch
        object itself. Whereas the branch object is used to hold all relevant
        power system parameters.

        Yields
        ------
        int
            Description #TODO check
        
        Note
        ----
        There are generator functions for nodes (`Graph.nodes()`) and edges
        (`Graph.edges()`) in NetworkX but unlike graph nodes, which can be
        represented by objects, branch objects can only be accessed by using an
        edge attribute ('branch' is used here)

        To make access to attributes of the branch objects simpler and more
        intuitive for the user, this generator yields a dictionary for each edge
        that contains information about adjacent nodes and the branch object.

        Note, the construction of the dictionary highly depends on the structure
        of the in-going tuple (which is defined by the needs of networkX). If
        this changes, the code will break.
        """

        # get edges with attributes
        edges = nx.get_edge_attributes(self._graph, 'branch').items()

        # sort them according to connected nodes
        edges_sorted = sorted(list(edges), key=lambda _: (''.join(sorted([repr(_[0][0]),repr(_[0][1])]))))

        for edge in edges_sorted:
            yield {'adj_nodes': edge[0], 'branch': edge[1]}

    def find_path(self, node_source, node_target, type='nodes'):
        """Determines shortest path

        Determines the shortest path from `node_source` to
        `node_target` in _graph using networkx' shortest path
        algorithm.

        Args
        ----
        node_source: GridDing0
            source node, member of _graph
        node_target: GridDing0
            target node, member of _graph
        type : str
            Specify if nodes or edges should be returned. Default
            is `nodes`

        Returns
        -------
        :any:`list` of :obj:`GridDing0`
            path: shortest path from `node_source` to `node_target` (list of nodes in _graph)

        Notes
        -----
        WARNING: The shortest path is calculated using the count of hops, not the actual line lengths!
        As long as the circuit breakers are open, this works fine since there's only one path. But if
        they are closed, there are 2 possible paths. The result is a path which have min. count of hops
        but might have a longer total path length than the second sone.
        See networkx' function shortest_path() function for details on how the path is calculated.
        """
        if (node_source in self._graph.nodes()) and (node_target in self._graph.nodes()):
            path = nx.shortest_path(self._graph, node_source, node_target)
        else:
            raise Exception('At least one of the nodes is not a member of graph.')
        if type == 'nodes':
            return path
        elif type == 'edges':
            return [_ for _ in self._graph.edges_iter(nbunch=path, data=True)
                    if (_[0] in path and _[1] in path)]
        else:
            raise ValueError('Please specify type as nodes or edges')

    def find_and_union_paths(self, node_source, nodes_target):
        """ Determines shortest paths from `node_source` to all nodes in `node_target` in _graph using find_path().
            
        The branches of all paths are stored in a set - the result is a list of unique branches.

        Args
        ----
        node_source: GridDing0 
            source node, member of _graph
        node_target: GridDing0
            target node, member of _graph

        Returns
        -------
        :any:`list` of :obj:`BranchDing0`
            branches: list of branches (list of nodes in _graph) #TODO:check
        """
        branches = set()
        for node_target in nodes_target:
            path = self.find_path(node_source, node_target)
            node_pairs = list(zip(path[0:len(path) - 1], path[1:len(path)]))
            for n1, n2 in node_pairs:
                branches.add(self._graph.edge[n1][n2]['branch'])

        return list(branches)

    def graph_path_length(self, node_source, node_target):
        """ Calculates the absolute distance between `node_source` and `node_target` in meters using find_path() and branches' length attribute.
            
        Args
        ----
        node_source: GridDing0 
            source node, member of _graph
        node_target: GridDing0
            target node, member of _graph

        Returns
        -------
        float
            path length in m
        """

        length = 0
        path = self.find_path(node_source, node_target)
        node_pairs = list(zip(path[0:len(path)-1], path[1:len(path)]))

        for n1, n2 in node_pairs:
            length += self._graph.edge[n1][n2]['branch'].length

        return length

    def graph_isolated_nodes(self):
        """ Finds isolated nodes = nodes with no neighbors (degree zero)

        Returns
        -------
        :any:`list` of :obj:`GridDing0`
            List of nodes (Ding0 objects)
        """
        return sorted(nx.isolates(self._graph), key=lambda x: repr(x))

    def control_generators(self, capacity_factor):
        """ Sets capacity factor of all generators of a grid.
        
        A capacity factor of 0.6 means that all generators are to provide a capacity of 60% of their nominal power.

        Args
        ----
        capacity_factor: float
            Value between 0 and 1.
        """

        for generator in self.generators():
            generator.capacity_factor = capacity_factor


class StationDing0:
    """
    Defines a HV-MV or MV-LV station in DINGO

    Parameters
    ----------
    id_db: :obj:`int`
        id according to database table
    v_level_operation: :obj:`float`
        operation voltage level at station (the station's voltage level differs from the nominal voltage level of 
        the grid due to grid losses).
        It is usually set to a slightly higher value than the nominal voltage, e.g. 104% in MV grids.
    geo_data : :shapely:`Shapely Point object<points>`
        Descr #TODO
    grid : :obj:`int`
        Desc #TODO
    busbar : :obj:`int`
        Desc #TODO
            
    See Also
    --------
     (see attribute `v_level` in class MVGridDing0) #TODO where is this?
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self._transformers = []
        self.v_level_operation = kwargs.get('v_level_operation', None)

    @property
    def network(self):
        return self.grid.network

    def transformers(self):
        """Returns a generator for iterating over transformers
        
        Yields
        ------
        StationDing0
            Description
        """
        for trans in self._transformers:
            yield trans

    def add_transformer(self, transformer):
        """Adds a transformer to _transformers if not already existing
        
        Args
        ----
        transformer : StationDing0
            Description #TODO
        """
        if transformer not in self.transformers() and isinstance(transformer, TransformerDing0):
            self._transformers.append(transformer)

    @property
    def peak_load(self):
        """
        Cumulative peak load of loads connected to underlying MV or LV grid
        
        (taken from MV or LV Grid District -> top-down)

        Notes
        -----
        This peak load includes all loads which are located within Grid District:
        When called from MV station, all loads of all Load Areas are considered
        (peak load was calculated in MVGridDistrictDing0.add_peak_demand()).
        When called from LV station, all loads of the LVGridDistrict are considered.
        """
        return self.grid.grid_district.peak_load


class RingDing0:
    """ Represents a medium voltage Ring
    """
    def __init__(self, **kwargs):
        self._grid = kwargs.get('grid', None)

        # get id from count of rings in associated MV grid
        self._id_db = self._grid.rings_count() + 1

        # add circ breaker to grid and graph
        self._grid.add_ring(self)

    @property
    def network(self):
        return self._grid.network

    def branches(self):
        """ #TODO: description
        """
        for branch in self._grid.graph_edges():
            if branch['branch'].ring == self:
                yield branch

    def lv_load_areas(self):
        """ #TODO: description
        """
        for lv_load_area in self._grid._graph.nodes():
            if isinstance(lv_load_area, LVLoadAreaDing0):
                if lv_load_area.ring == self:
                    yield lv_load_area

    def __repr__(self):
        return 'mv_ring_' + str(self._id_db)


class BranchDing0:
    # TODO: check docstring
    """ #TODO Description of Class
    
    Attributes
    ----------
    length : :obj:`float`
        Length of line given in m
    type : :pandas:`pandas.DataFrame<dataframe>`
        Association to pandas Series. DataFrame with attributes of line/cable.
    id_db : :obj:`int`
        id according to database table
    ring : :obj:`int`
        Description #TODO
    kind : :obj:`str`
        'line' or 'cable'
    connects_aggregated : 
        Description #TODO
    circuit_breaker : :obj:`CircuitBreakerDing0`
        Description #TODO

    Notes
    -----
    Important: id_db is not set until whole grid is finished (setting at the end).
        
    See Also
    --------
    ding0.core.network.grids.MVGridDing0.set_branch_ids
    """

    def __init__(self, **kwargs):

        self.id_db = kwargs.get('id_db', None)
        self.ring = kwargs.get('ring', None)
        self.length = kwargs.get('length', None)  # branch (line/cable) length in m
        self.kind = kwargs.get('kind', None)  # 'line' or 'cable'
        self.type = kwargs.get('type', None)  # DataFrame with attributes of line/cable
        self.connects_aggregated = kwargs.get('connects_aggregated', False)
        self.circuit_breaker = kwargs.get('circuit_breaker', None)

        self.critical = False

    @property
    def network(self):
        return self.ring.network

    def __repr__(self):
        return 'branch_' + str(self.id_db)


class TransformerDing0:
    """ #TODO description Transformers
    
    Attributes
    ----------
    id_db : :obj:`int`
        id according to database table
    v_level : :obj:`float`
        voltage level	
    s_max_a : :obj:`float`
        rated power (long term)	
    s_max_b : :obj:`float`
        rated power (short term)	        
    s_max_c : :obj:`float`
        rated power (emergency)	
    phase_angle : :obj:`float`
        phase shift angle
    tap_ratio: :obj:`float`
        off nominal turns ratio
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.grid = kwargs.get('grid', None)
        self.v_level = kwargs.get('v_level', None)
        self.s_max_a = kwargs.get('s_max_longterm', None)
        self.s_max_b = kwargs.get('s_max_shortterm', None)
        self.s_max_c = kwargs.get('s_max_emergency', None)
        self.phase_angle = kwargs.get('phase_angle', None)
        self.tap_ratio = kwargs.get('tap_ratio', None)
        self.r = kwargs.get('r', None)
        self.x = kwargs.get('x', None)

    @property
    def network(self):
        return self.grid.network


class GeneratorDing0:
    """ Generators (power plants of any kind)
        
    Attributes
    ----------
    id_db : :obj:`int`
        id according to database table
    name : :obj:`str`
        Description #TODO
    v_level : :obj:`float`
        voltage level
    geo_data : :shapely:`Shapely Point object<points>`
        Descr #TODO
    mv_grid : :obj:`int`
        Descr #TODO
    lv_load_area : :obj:`int`
        Descr #TODO
    lv_grid : :obj:`int`
        Descr #TODO
    capacity : :obj:`float`
        Descr #TODO
    capacity_factor : :obj:`float`
        Descr #TODO
    type : :obj:`int`
        Descr #TODO
    subtype : :obj:`int`
        Descr #TODO
        
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.name = kwargs.get('name', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.mv_grid = kwargs.get('mv_grid', None)
        self.lv_load_area = kwargs.get('lv_load_area', None)
        self.lv_grid = kwargs.get('lv_grid', None)

        self.capacity = kwargs.get('capacity', None)
        self.capacity_factor = kwargs.get('capacity_factor', 1)
        self.type = kwargs.get('type', None)
        self.subtype = kwargs.get('subtype', None)
        self.v_level = kwargs.get('v_level', None)

    @property
    def network(self):
        return self.mv_grid.network

    @property
    def pypsa_id(self):
        return '_'.join(['MV', str(self.mv_grid.id_db),
                                  'gen', str(self.id_db)])

    def __repr__(self):
        if self.v_level in ['6', '7']:
            return ('generator_' + str(self.type) + '_' + str(self.subtype) +
                    '_mvgd' + str(self.mv_grid.grid_district.id_db) +
                    '_lvgd' + str(self.lv_grid.id_db) + '_' + str(self.id_db))
        else:
            return ('generator_' + str(self.type) + '_' + str(self.subtype) +
                    '_mvgd' + str(self.mv_grid.id_db) + '_' + str(self.id_db))


class GeneratorFluctuatingDing0(GeneratorDing0):
    """Generator object for fluctuating renewable energy sources

    Attributes
    ----------
    _weather_cell_id : :obj:`str`
        ID of the weather cell used to generate feed-in time series

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._weather_cell_id = kwargs.get('weather_cell_id', None)

    @property
    def weather_cell_id(self):
        """
        Get weather cell ID
        Returns
        -------
        :obj:`str`
            See class definition for details.
        """
        return self._weather_cell_id

    @weather_cell_id.setter
    def weather_cell_id(self, weather_cell):
        self._weather_cell_id = weather_cell


class CableDistributorDing0:
    """ Cable distributor (connection point) 
     
    Attributes
    ----------
    id_db : :obj:`int`
        id according to database table
    geo_data : :shapely:`Shapely Point object<points>`
        Descr #TODO
    grid : :obj:`int`
        Descr #TODO
    
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)

    @property
    def network(self):
        return self.grid.network


class LoadDing0:
    """ Class for modelling a load 
        
    Attributes
    ----------
    id_db : :obj:`int`
        id according to database table
    geo_data : :shapely:`Shapely Point object<points>`
        Descr #TODO
    grid : :obj:`int`
        Descr #TODO
    peak_load : :obj:`float`
        Descr #TODO
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self.peak_load = kwargs.get('peak_load', None)
        self.consumption = kwargs.get('consumption', None)

        self.id_db = self.grid.loads_count() + 1

    @property
    def network(self):
        return self.grid.network


class CircuitBreakerDing0:
    """ Class for modelling a circuit breaker

    Attributes
    ----------
    id_db : :obj:`int`
        id according to database table
    geo_data : :shapely:`Shapely Point object<points>`
        Descr #TODO
    grid : :obj:`int`
        Descr #TODO
    branch : :obj:`BranchDing0`
        Descr #TODO
    branch_nodes : (:obj:`GridDing0`, :obj:`GridDing0`)
        Tuple of nodes in the branch.
    status: :obj:`str`, default 'closed'
        Desc #TODO
        
    Notes
    -----
    Circuit breakers are nodes of a graph, but are NOT connected via an edge. They are associated to a specific
    `branch` of a graph (and the branch refers to the circuit breaker via the attribute `circuit_breaker`) and its
    two `branch_nodes`. Via open() and close() the associated branch can be removed from or added to graph.

    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self.branch = kwargs.get('branch', None)
        self.branch_nodes = kwargs.get('branch_nodes', (None, None))
        self.status = kwargs.get('status', 'closed')

        # get id from count of cable distributors in associated MV grid
        self.id_db = self.grid.circuit_breakers_count() + 1

        # add circ breaker to grid and graph
        self.grid.add_circuit_breaker(self)

    @property
    def network(self):
        """ #Todo Description
        """
        return self.grid.network

    def open(self):
        """ Open a Circuit Breaker #TODO Check
        """
        self.branch_nodes = self.grid.graph_nodes_from_branch(self.branch)
        self.grid._graph.remove_edge(self.branch_nodes[0], self.branch_nodes[1])
        self.status = 'open'

    def close(self):
        """ Close a Circuit Breaker #TODO Check
        """
        self.grid._graph.add_edge(self.branch_nodes[0], self.branch_nodes[1], branch=self.branch)
        self.status = 'closed'

    def __repr__(self):
        return 'circuit_breaker_' + str(self.id_db)
