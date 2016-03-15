from oemof.core.network import Entity
from oemof.core.network.entities.components import Transformer
from oemof.core.network.entities.components import Transport
from oemof.core.network.entities.components import Source
from oemof.core.network.entities.buses import Bus

from dingo.core.structure.regions import LVRegionDingo

#from oemof.core.network.entities.buses import BusPypo
#from oemof.core.network.entities.components.transports import BranchPypo
#from oemof.core.network.entities.components.sources import GenPypo

import networkx as nx
import matplotlib.pyplot as plt

class GridDingo():
    """ DINGO grid

    id_db: id according to database table
    """
    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        #self.geo_data = kwargs.get('geo_data', None)
        self.region = kwargs.get('region', None)

        self.graph = nx.Graph()

    # TODO: UPDATE DRAW FUNCTION -> make draw method work for both MV and LV regions!
    def draw(self):
        """draws grid graph using networkx"""

        g = self.graph

        # get node positions
        nodes_pos = {}
        #x=nd._mv_regions[0].mv_grid.graph.nodes()
        #x = self.graph.nodes()
        #x = self.region._lv_regions

        #lv_regions = [_ for _ in self.graph.nodes() if isinstance(_, LVRegionDingo)]

        for node in self.region._lv_regions:
            nodes_pos[node] = tuple(node.geo_surfacepnt)
        for node in self.region.
            nodes_pos[node] = tuple

        ntemp = []
        nodes_pos = {}
        demands = {}
        demands_pos = {}
        for no, node in self._nodes.items():
            g.add_node(node)
            ntemp.append(node)
            coord = self._problem._coord[no]
            nodes_pos[node] = tuple(coord)
            demands[node] = 'd=' + str(node.demand())
            demands_pos[node] = tuple([a+b for a, b in zip(coord, [2.5]*len(coord))])

        for r in self.routes():
            #print(r)
            n1 = r._nodes[0:len(r._nodes)-1]
            n2 = r._nodes[1:len(r._nodes)]
            e = list(zip(n1, n2))
            depot = self._nodes[1]
            e.append((depot, r._nodes[0]))
            e.append((r._nodes[-1], depot))
            g.add_edges_from(e)

        plt.figure()
        nx.draw_networkx(g, nodes_pos)
        nx.draw_networkx_labels(g, demands_pos, labels=demands)
        plt.show()

    def draw_networkx(self, graph):
        # TODO: add node-specific attributes (e.g. color) to ensure nice output when plotting (e.g. different colors for
        # MV and LV stations

        positions = {}
        #positions_b = {}
        for t in self.transformers:
            positions[t] = [t.geo_data.x, t.geo_data.y]
        #for b in self.buses:
        #    positions_b[b] = [b.geo_data.x, b.geo_data.y]
        #positions[] = [[t.geo_data.x, t.geo_data.y] for t in self.transformers]
        nx.draw_networkx_nodes(graph, positions, self.transformers, node_shape="o", node_color="r",
                               node_size = 5)
        #nx.draw_networkx_nodes(graph, positions_b, self.buses, node_shape="o", node_color="b",
        #                       node_size = 2
        plt.show()

#        for e in entities:
#            for e_in in e.inputs:
#                g.add_edge(e_in, e)
#        if positions is None:
#            positions = nx.spectral_layout(g)
#        nx.draw_networkx_nodes(g, positions, buses, node_shape="o", node_color="r",
#                               node_size = 600)
#        nx.draw_networkx_nodes(g, positions, branches, node_shape="s",
#                               node_color="b", node_size=200)
#        nx.draw_networkx_nodes(g, positions, generators, node_shape="s",
#                               node_color="g", node_size=200)
#        nx.draw_networkx_edges(g, positions)
#        if labels:
#            nx.draw_networkx_labels(g, positions)
#        plt.show()
#        if not nx.is_connected(g):
#            raise ValueError("Graph is not connected")

    #def import_mv_stations(self, conn, id=None):


class StationDingo():
    """
    Defines a MV/LVstation in DINGO
    -------------------------------

    id_db: id according to database table
    """

    def __init__(self, **kwargs):
        self.id_db = kwargs.get('id_db', None)
        self.geo_data = kwargs.get('geo_data', None)
        self.grid = kwargs.get('grid', None)
        self.transformers = kwargs.get('transformers', None)
        self.busbar = None

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
        

class BranchDingo(Transport):
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
        #inherit parameters from oemof's Transport
        super().__init__(**kwargs)
        #more params
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
