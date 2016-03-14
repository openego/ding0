from dingo.core.network.stations import *
from dingo.core.structure.regions import *
from dingo.tools import config as cfg_dingo
from oemof import db

import networkx as nx
import pandas as pd
from geopy.distance import vincenty

class NetworkDingo():
    """ DINGO Network, contains the NetworkX graph and associated attributes.
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', None)
        self.mv_region = kwargs.get('mv_region', None)

        self.graph = nx.Graph()
        self.graph

    def convert_to_networkx(self):
        #entities = self.entities
        #buses = [e for e in entities if isinstance(e, BusPypo)]
        #generators = [e for e in entities if isinstance(e, GenPypo)]
        #branches = [e for e in entities if isinstance(e, BranchPypo)]
        self.graph.add_nodes_from(self.transformers)
        #graph.add_nodes_from(self.transformers + self.buses)
        #positions = nx.spectral_layout(graph)
        return graph

    def draw_networkx(self, graph):
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
    def import_mv_stations(self, id=None):
        """imports MV-stations from database"""

        conn = db.connection(section='ontohub_wdb_remote')

        schema_table = cfg_dingo.get('stations', 'mv_stations')
        #schema, table = cfg_dingo.get('stations', 'mv_stations').split('.')
        index_col = 'subst_id'
        #columns = ['subst_id', 'geom']
        srid = '4326'

        # build SQL query
        where_clause = ''
        if id:
            where_clause =  'WHERE subst_id=' + str(id)
        sql = """SELECT subst_id,
                        ST_AsText(ST_TRANSFORM(geom, {0})) as geom
                        FROM {1} {2};""".format(srid, schema_table, where_clause)

        # read data from db
        mv_stations = pd.read_sql_query(sql, conn, index_col)

        # create station objects from rows and add them to graph
        for idx, row in mv_stations.iterrows():
            station_obj = MVStationDingo(name='USW_'+str(idx), geo_data=row['geom'])
            self.graph.add_node(station_obj)

        conn.close()

    #def import_lv_regions(self, conn):
    def import_lv_regions(self):
        """imports LV-regions (load ares) from database

        Table definition for load areas can be found here:
        http://vernetzen.uni-flensburg.de/redmine/projects/open_ego/wiki/Methoden_AP_26_DataProc
        """

        conn = db.connection(section='ontohub_oedb_remote')

        schema_table = cfg_dingo.get('regions', 'lv_regions')
        index_col = 'lgid'
        srid = '4326'

        # build SQL query
        # TODO: insert WHERE-statement here according to Lui's upcoming table definition. Pseudo-Code: SELECT stuff FROM all LV-regions which are within MV-Region (Polygon) x
        where_clause = ''

        sql = """SELECT lgid,
                        zensus_sum,
                        zensus_count,
                        zensus_density,
                        ioer_sum,
                        ioer_count,
                        ioer_density,
                        area_lg,
                        sector_area_residential,
                        sector_area_retail,
                        sector_area_industrial,
                        sector_area_agricultural,
                        sector_share_residential,
                        sector_share_retail,
                        sector_share_industrial,
                        sector_share_agricultural,
                        sector_count_residential,
                        sector_count_retail,
                        sector_count_industrial,
                        sector_count_agricultural,
                        sector_consumption_residential,
                        sector_consumption_retail,
                        sector_consumption_industrial,
                        sector_consumption_agricultural,
                        nuts,
                        ST_AsText(ST_TRANSFORM(geom, {0})) as geom_area,
                        ST_AsText(ST_TRANSFORM(geom_centroid, {0})) as geom_centroid,
                        ST_AsText(ST_TRANSFORM(geom_surfacepoint, {0})) as geom_surfacepoint
                 FROM {1} {2};""".format(srid, schema_table, where_clause)

        # read data from db
        lv_regions = pd.read_sql_query(sql, conn, index_col)

        # create region objects from rows and add them to graph
        for idx, row in lv_regions.iterrows():
            region_obj = LVRegionDingo(db_data=row, mv_region=)#, db_cols=lv_regions.columns.values)
            #self.graph.add_node(station_obj)

        conn.close()