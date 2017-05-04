import matplotlib.pyplot as plt
import networkx as nx

from dingo.core.structure.regions import LVLoadAreaDingo, LVLoadAreaCentreDingo


class GridDingo:
    """ DINGO grid

    Parameters
    ----------
    id_db : id according to database table
    grid_district: class, area that is covered by the lv grid
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
        """Returns a generator for iterating over cable distributors"""
        for cable_dist in self._cable_distributors:
            yield cable_dist

    def cable_distributors_count(self):
        """Returns the count of cable distributors in grid"""
        return len(self._cable_distributors)

    def loads(self):
        """Returns a generator for iterating over grid's loads"""
        for load in self._loads:
            yield load

    def loads_count(self):
        """Returns the count of loads in grid"""
        return len(self._loads)

    def generators(self):
        """Returns a generator for iterating over grid's generators"""
        for generator in self._generators:
            yield generator

    def add_generator(self, generator):
        """Adds a generator to _generators and grid graph if not already existing"""
        if generator not in self._generators and isinstance(generator,
                                                            GeneratorDingo):
            self._generators.append(generator)
            self.graph_add_node(generator)

    def graph_add_node(self, node_object):
        """Adds a station or cable distributor object to grid graph if not already existing"""
        if ((node_object not in self._graph.nodes()) and
            (isinstance(node_object, (StationDingo,
                                      CableDistributorDingo,
                                      LVLoadAreaCentreDingo,
                                      CircuitBreakerDingo,
                                      GeneratorDingo)))):
            self._graph.add_node(node_object)

    # TODO: UPDATE DRAW FUNCTION -> make draw method work for both MV and load_areas!
    def graph_draw(self):
        """ Draws grid graph using networkx

        caution: The geo coords (for used crs see database import in class `NetworkDingo`) are used as positions for
                 drawing but networkx uses cartesian crs. Since no coordinate transformation is performed, the drawn
                 graph representation is falsified!
        """

        g = self._graph

        # get draw params from nodes and edges (coordinates, colors, demands, etc.)
        nodes_pos = {}; demands = {}; demands_pos = {}
        nodes_color = []
        for node in g.nodes():
            if isinstance(node, (StationDingo,
                                 LVLoadAreaCentreDingo,
                                 CableDistributorDingo,
                                 GeneratorDingo,
                                 CircuitBreakerDingo)):
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

    def graph_nodes_sorted(self):
        """ Returns an (ascending) sorted list of graph's nodes (name is used as key).

        """
        return sorted(self._graph.nodes(), key=lambda _: repr(_))

    def graph_nodes_from_branch(self, branch):
        """ Returns nodes that are connected by `branch`

        Args:
            branch: BranchDingo object
        Returns:
            2-tuple of nodes (Dingo objects)
        """
        edges = nx.get_edge_attributes(self._graph, 'branch')
        nodes = list(edges.keys())[list(edges.values()).index(branch)]
        return nodes

    def graph_branches_from_node(self, node):
        """ Returns branches that are connected to `node`

        Args:
            node: Dingo object (member of graph)
        Returns:
            branches: List of tuples (node, branch), content: node=Dingo object (member of graph),
                                                              branch=BranchDingo object
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
        edges_sorted = sorted(list(edges), key=lambda _: repr(_[0]))

        for edge in edges_sorted:
            yield {'adj_nodes': edge[0], 'branch': edge[1]}

    def find_path(self, node_source, node_target):
        """ Determines the shortest path from `node_source` to `node_target` in _graph using networkx' shortest path
            algorithm.

        Args:
            node_source: source node (Dingo object), member of _graph
            node_target: target node (Dingo object), member of _graph

        Returns:
            path: shortest path from `node_source` to `node_target` (list of nodes in _graph)
        """
        if (node_source in self._graph.nodes()) and (node_target in self._graph.nodes()):
            path = nx.shortest_path(self._graph, node_source, node_target)
        else:
            raise Exception('At least one of the nodes is not a member of graph.')

        return path

    def find_and_union_paths(self, node_source, nodes_target):
        """ Determines shortest paths from `node_source` to all nodes in `node_target` in _graph using find_path().
            The branches of all paths are stored in a set - the result is a list of unique branches.

        Args:
            node_source: source node (Dingo object), member of _graph
            nodes_target: list of target nodes (Dingo objects), members of _graph

        Returns:
            branches: list of branches (list of nodes in _graph)
        """
        branches = set()
        for node_target in nodes_target:
            path = self.find_path(node_source, node_target)
            node_pairs = list(zip(path[0:len(path) - 1], path[1:len(path)]))
            for n1, n2 in node_pairs:
                branches.add(self._graph.edge[n1][n2]['branch'])

        return list(branches)

    def graph_path_length(self, node_source, node_target):
        """ Calculates the absolute distance between `node_source` and `node_target` in meters using find_path() and
            branches' length attribute.
        Args:
            node_source: source node (Dingo object), member of _graph
            node_target: target node (Dingo object), member of _graph

        Returns:
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
        Args:
            none
        Returns:
            List of nodes (Dingo objects)
        """
        return sorted(nx.isolates(self._graph), key=lambda x: repr(x))

    def control_generators(self, capacity_factor):
        """ Sets capacity factor of all generators of a grid (example: A capacity factor of 0.6 means that all
            generators are to provide a capacity of 60% of their nominal power).

        Args:
            capacity_factor: 0..1
        """

        for generator in self.generators():
            generator.capacity_factor = capacity_factor


class StationDingo:
    """
    Defines a MV/LVstation in DINGO
    -------------------------------

    id_db: id according to database table
    v_level_operation: operation voltage level at station (the station's voltage level differs from the nominal voltage
                       level of the grid (see attribute `v_level` in class MVGridDingo) due to grid losses. It is
                       usually set to a slightly higher value than the nominal voltage, e.g. 104% in MV grids.

    """
    # TODO: add method remove_transformer()

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self._transformers = []
        self.busbar = None
        self.v_level_operation = kwargs.get('v_level_operation', None)

    def transformers(self):
        """Returns a generator for iterating over transformers"""
        for trans in self._transformers:
            yield trans

    def add_transformer(self, transformer):
        """Adds a transformer to _transformers if not already existing"""
        # TODO: check arg
        if transformer not in self.transformers() and isinstance(transformer, TransformerDingo):
            self._transformers.append(transformer)
        # TODO: what if it exists? -> error message

    @property
    def peak_load(self):
        """
        Cumulative peak load of loads connected to underlying LV grid
        """
        return self.grid.grid_district.peak_load_sum
        #return sum([_.peak_load for _ in self.grid.loads()])

    @property
    def peak_generation(self):
        """
        Cumulative peak generation of generators connected to underlying LV grid
        (instantaneously calculated -> bottom-up)
        """
        return sum([_.capacity for _ in self.grid.generators()])


class BusDingo:
    """ Create new pypower Bus class as child from oemof Bus used to define
    busses and generators data
    """

    def __init__(self, **kwargs):
        """Assigned minimal required pypower input parameters of the bus and
        generator as arguments

        Keyword description of bus arguments:
        bus_id -- the bus number (also used as GEN_BUS parameter for generator)
        bus_type -- the bus type (1 = PQ, 2 = PV, 3 = ref, 4 = Isolated)
        PD -- the real power demand in MW
        QD -- the reactive power demand in MVAr
        GS -- the shunt conductance (demanded at V = 1.0 p.u.) in MW
        BS -- the shunt susceptance (injected at V = 1.0 p.u.) in MVAr
        bus_area -- area number (positive integer)
        VM -- the voltage magnitude in p.u.
        VA -- the voltage angle in degrees
        base_kv -- the base voltage in kV
        zone -- loss zone (positive integer)
        vmax -- the maximum allowed voltage magnitude in p.u.
        vmin -- the minimum allowed voltage magnitude in p.u.
        """

        # Bus Data parameters


class RingDingo:
    """ Represents a medium voltage Ring
    """
    def __init__(self, **kwargs):
        self._grid = kwargs.get('grid', None)

        # get id from count of rings in associated MV grid
        self._id_db = self._grid.rings_count() + 1

        # add circ breaker to grid and graph
        self._grid.add_ring(self)

    def branches(self):
        for branch in self._grid.graph_edges():
            if branch['branch'].ring == self:
                yield branch

    def lv_load_areas(self):
        for lv_load_area in self._grid._graph.nodes():
            if isinstance(lv_load_area, LVLoadAreaDingo):
                if lv_load_area.ring == self:
                    yield lv_load_area

    def __repr__(self):
        return 'mv_ring_' + str(self._id_db)


class BranchDingo:
    """
    Parameter
    ----------------
    length : float
        Length of line given
    type : Association to pandas Series

    Notes:
        Important: id_db is not set until whole grid is finished (setting at the end, see method set_branch_ids())
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

    def __repr__(self):
        return 'branch_' + str(self.id_db)


class TransformerDingo:
    """
    Transformers
    ------------
    id_db : int
        id according to database table
    v_level : 
        voltage level	
    s_max_a : float
        rated power (long term)	
    s_max_b : float
        rated power (short term)	        
    s_max_c : float
        rated power (emergency)	
    phase_angle : float
        phase shift angle
    tap_ratio: float
        off nominal turns ratio
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.v_level = kwargs.get('v_level', None)
        self.s_max_a = kwargs.get('s_max_longterm', None)
        self.s_max_b = kwargs.get('s_max_shortterm', None)
        self.s_max_c = kwargs.get('s_max_emergency', None)
        self.phase_angle = kwargs.get('phase_angle', None)
        self.tap_ratio = kwargs.get('tap_ratio', None)
        self.r = kwargs.get('r', None)
        self.x = kwargs.get('x', None)


class GeneratorDingo:
    """ Generators (power plants of any kind)
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


class CableDistributorDingo:
    """ Cable distributor (connection point) """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)


class LoadDingo:
    """ Class for modelling a load """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self.peak_load = kwargs.get('peak_load', None)

        self.id_db = self.grid.loads_count() + 1


class CircuitBreakerDingo:
    """ Class for modelling a circuit breaker

    Notes:
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

    def open(self):
        self.branch_nodes = self.grid.graph_nodes_from_branch(self.branch)
        self.grid._graph.remove_edge(self.branch_nodes[0], self.branch_nodes[1])
        self.status = 'open'

    def close(self):
        self.grid._graph.add_edge(self.branch_nodes[0], self.branch_nodes[1], branch=self.branch)
        self.status = 'closed'

    def __repr__(self):
        return 'circuit_breaker_' + str(self.id_db)
