"""
Graph processing.
"""
import networkx as nx
from geoalchemy2.shape import to_shape
from config.config_lv_grids_osm import get_config_osm 


def update_ways_geo_to_shape(ways_sql_df):
    """
    update_ways_geo_to_shape
    after loading from DB
    """

    ways_sql_df['geometry'] = ways_sql_df.apply(lambda way: to_shape(way.geometry).coords, axis=1)

    return ways_sql_df


def build_graph_from_ways(ways):

    """ 
    Build graph based on osm ways
    Based on this graph, the routing will be implemented.
    """
    # init a graph and set srid.
    graph = nx.MultiGraph()
    graph.graph["crs"] = 'epsg:'+str(get_config_osm('srid')) 

    for w in ways:
        # add edges
        ix = 0
        for node in w.nodes[:-1]:
            # add_edge(u,v,geometry=line,length=leng,highway=highway,osmid=osmid)
            graph.add_edge(w.nodes[ix], w.nodes[ix+1], length=w.length_segments[ix], 
                                  osmid=w.osm_id, highway=w.highway)
            ix += 1

        # add coords
        ix = 0
        coords_list = list(to_shape(w.geometry).coords)
        
        for node in w.nodes:
            graph.nodes[node]['x'] = coords_list[ix][0]
            graph.nodes[node]['y'] = coords_list[ix][1]
            graph.nodes[node]['node_type'] = 'non_synthetic'

            ix += 1

    # convert undirected graph to digraph due to truncate_graph_polygon
    graph = graph.to_directed()  
    
    return graph
