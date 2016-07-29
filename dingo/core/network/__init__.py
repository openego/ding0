from oemof.core.network.entities.components import Source
from oemof.core.network.entities.buses import Bus

from dingo.core.structure.regions import LVLoadAreaCentreDingo

#from oemof.core.network.entities.buses import BusPypo
#from oemof.core.network.entities.components.transports import BranchPypo
#from oemof.core.network.entities.components.sources import GenPypo

import networkx as nx
import matplotlib.pyplot as plt


class GridDingo:
    """ DINGO grid

    Parameters
    ----------
    id_db : id according to database table
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.grid_district = kwargs.get('region', None)
        #self.geo_data = kwargs.get('geo_data', None)
        self.v_level = kwargs.get('v_level', None)

        self._graph = nx.Graph()

    def graph_add_node(self, node_object):
        """Adds a station or cable distributor object to grid graph if not already existing"""
        if ((node_object not in self._graph.nodes()) and
            (isinstance(node_object, (StationDingo, CableDistributorDingo, LVLoadAreaCentreDingo)))):
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
            if isinstance(node, (StationDingo, LVLoadAreaCentreDingo, CableDistributorDingo)):
                nodes_pos[node] = (node.geo_data.x, node.geo_data.y)
                # TODO: MOVE draw/color settings to config
            if node == self.station():
                nodes_color.append((1, 0.5, 0.5))
            else:
                #demands[node] = 'd=' + '{:.3f}'.format(node.grid.region.peak_load_sum)
                #demands_pos[node] = tuple([a+b for a, b in zip(nodes_pos[node], [0.003]*len(nodes_pos[node]))])
                nodes_color.append((0.5, 0.5, 1))

        plt.figure()
        nx.draw_networkx(g, nodes_pos, node_color=nodes_color, font_size=10)
        nx.draw_networkx_labels(g, demands_pos, labels=demands, font_size=8)
        plt.show()

    def graph_nodes_sorted(self):
        """ Returns an (ascending) sorted list of graph's nodes (name is used as key).

        """
        return sorted(self._graph.nodes(), key=lambda _: repr(_))

    def graph_edges(self):
        """ Returns a generator for iterating over graph edges

        The edge of a graph is described by the to adjacent node and the branch
        object itself. Whereas the branch object is used to hold all relevant
        power system parameters.

        Note
        ----

        There are generator functions for nodes (`Graph.nodes()`) and edges
        (`Graph.edges()`) in NetworkX but unlike graph nodes, which can be
        represented by objects, branch objects can only be accessed by using an
        edge attribute ('branch' is used here)

        To make access to attributes of the branch objects simplier and more
        intuitive for the user, this generator yields a dictionary for each edge
        that contains information about adjacent nodes and the branch object.

        Note, the construction of the dictionary highly depends on the structure
        of the in-going tuple (which is defined by the needs of networkX). If
        this changes, the code will break.
        """
        for edge in nx.get_edge_attributes(self._graph, 'branch').items():
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


class StationDingo():
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
        self.peak_load = kwargs.get('peak_load', None)
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


class BusDingo(Bus):
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

        super().__init__(**kwargs)
        # Bus Data parameters
        

class BranchDingo:
    """
    Parameter
    ----------------
    length : float
        Length of line given
    type : Association to pandas Series
    """


    def __init__(self, **kwargs):
        # inherit parameters from oemof's Transport
        #super().__init__(**kwargs)

        # branch (line/cable) length in m
        self.length = kwargs.get('length', None)
        self.type = kwargs.get('type', None)
<<<<<<< HEAD
=======


>>>>>>> e547894... Adapt class BranchDingo to it's actual needs


class TransformerDingo():
    """
    Transformers
    ------------
    geo_data : shapely.geometry object
        Geo-spatial data with informations for location/region-shape. The
        geometry can be a polygon/multi-polygon for regions, a line for
        transport objects or a point for objects such as transformer sources.
    equip_trans_id : int
        ID of transformer type according to DB table 'equip_trans'
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
        #inherit parameters from oemof's Transformer
        # super().__init__(**kwargs)
        #more params
        self.equip_trans_id = kwargs.get('equip_trans_id', None)
        self.v_level = kwargs.get('v_level', None)
        self.s_max_a = kwargs.get('s_max_longterm', None)
        self.s_max_b = kwargs.get('s_max_shortterm', None)
        self.s_max_c = kwargs.get('s_max_emergency', None)
        self.phase_angle = kwargs.get('phase_angle', None)
        self.tap_ratio = kwargs.get('tap_ratio', None)


class SourceDingo(Source):
    """
    Generators
    """

    def __init__(self, **kwargs):
        #inherit parameters from oemof's Transformer
        super().__init__(**kwargs)


class CableDistributorDingo():
    """ Cable distributor (connection point) """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self.lv_load_area_group = kwargs.get('lv_load_area_group', None)

        # get id from count of cable distributors in associated MV grid
        self.id_db = self.grid.cable_distributors_count() + 1

    def __repr__(self):
        return 'cable_dist_' + str(self.id_db)
