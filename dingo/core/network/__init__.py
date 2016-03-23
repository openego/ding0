from oemof.core.network import Entity
from oemof.core.network.entities.components import Transformer
from oemof.core.network.entities.components import Transport
from oemof.core.network.entities.components import Source
from oemof.core.network.entities.buses import Bus

#from oemof.core.network.entities.buses import BusPypo
#from oemof.core.network.entities.components.transports import BranchPypo
#from oemof.core.network.entities.components.sources import GenPypo

import networkx as nx
import matplotlib.pyplot as plt


class GridDingo:
    """ DINGO grid

    id_db: id according to database table
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        #self.geo_data = kwargs.get('geo_data', None)
        self.region = kwargs.get('region', None)
        #self._stations = []

        self._graph = nx.Graph()

    def graph_add_node(self, station):
        """Adds a station object to grid graph if not already existing"""
        if station not in self._graph.nodes() and isinstance(station, StationDingo):
            self._graph.add_node(station)

    # TODO: UPDATE DRAW FUNCTION -> make draw method work for both MV and LV regions!
    def graph_draw(self):
        """draws grid graph using networkx

        caution: The geo coords (for used crs see database import in class `NetworkDingo`) are used as positions for
                 drawing but networkx uses cartesian crs. Since no coordinate transformation is performed, the drawn
                 graph representation is falsified!
        """

        g = self._graph

        # get draw params from nodes and edges (coordinates, colors, demands, etc.)
        nodes_pos = {}; demands = {}; demands_pos = {}
        nodes_color = []
        for node in g.nodes():
            if isinstance(node, StationDingo):
                nodes_pos[node] = (node.geo_data.x, node.geo_data.y)
                # TODO: Add demand as label
            if node == self.station():
                nodes_color.append((1, 0.5, 0.5))
            else:
                demands[node] = 'd=' + '{:.3f}'.format(node.grid.region.peak_load_sum)
                demands_pos[node] = tuple([a+b for a, b in zip(nodes_pos[node], [0.002]*len(nodes_pos[node]))])
                nodes_color.append((0.5, 0.5, 1))

        plt.figure()
        nx.draw_networkx(g, nodes_pos, node_color=nodes_color, font_size=10)
        nx.draw_networkx_labels(g, demands_pos, labels=demands, font_size=8)
        plt.show()


class StationDingo():
    """
    Defines a MV/LVstation in DINGO
    -------------------------------

    id_db: id according to database table
    """
    # TODO: add method remove_transformer()

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self._transformers = []
        self.busbar = None

    def transformers(self):
        """Returns a generator for iterating over transformers"""
        for trans in self._transformers:
            yield trans

    def add_transformer(self, transformer):
        """Adds a transformer to _transformers if not already existing"""
        # TODO: check arg
        if transformer not in self.transformers() and isinstance(transformer, TransformerDingo):
            self._transformers.append(transformer)


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
        

class BranchDingo():
    """
    Cables and lines
    ----------------
    geo_data : shapely.geometry object
        Geo-spatial data with informations for location/region-shape. The
        geometry can be a polygon/multi-polygon for regions, a line for
        transport objects or a point for objects such as transformer sources.
    equip_line_id : int
        ID of cable/line type according to DB table 'equip_line'
    out_max : float
        Maximum output which can possibly be obtained when using the transport,
        in $MW$.
    """


    def __init__(self, **kwargs):
        # inherit parameters from oemof's Transport
        #super().__init__(**kwargs)

        #

        # more params (OLD)
        self.equip_line_id = kwargs.get('equip_line_id', None)
        self.v_level = kwargs.get('v_level', None)
        self.type = kwargs.get('type', None)
        self.cable_cnt = kwargs.get('cable_cnt', None)
        self.wire_cnt = kwargs.get('wire_cnt', None)
        self.cs_area = kwargs.get('cs_area', None)
        self.r = kwargs.get('r', None)
        self.x = kwargs.get('x', None)
        self.c = kwargs.get('c', None)
        self.i_max_th = kwargs.get('i_max_th', None)
        self.s_max_a = kwargs.get('s_max_a', None)
        self.s_max_b = kwargs.get('s_max_b', None)
        self.s_max_c = kwargs.get('s_max_c', None)



class TransformerDingo(Transformer):
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
        super().__init__(**kwargs)
        #more params
        self.equip_trans_id = kwargs.get('equip_trans_id', None)
        self.v_level = kwargs.get('v_level', None)
        self.s_max_a = kwargs.get('s_max_a', None)
        self.s_max_b = kwargs.get('s_max_b', None)
        self.s_max_c = kwargs.get('s_max_c', None)
        self.phase_angle = kwargs.get('phase_angle', None)
        self.tap_ratio = kwargs.get('tap_ratio', None)


class SourceDingo(Source):
    """
    Generators
    """

    def __init__(self, **kwargs):
        #inherit parameters from oemof's Transformer
        super().__init__(**kwargs)
