"""
Routing based on ways from OSM.
TODO: Separate routing.py to graph_processing.py
"""

import osmnx as ox
import networkx as nx

#from pyproj import CRS

import pandas as pd
#import geopandas as gpd

import numpy as np

#from itertools import combinations

from shapely.geometry import LineString, Point #, MultiLineString, Polygon # PAUL NEW
#from shapely.ops import linemerge
#from shapely.wkt import dumps as wkt_dumps

from ding0.config.config_lv_grids_osm import get_config_osm

#from ding0.grid.lv_grid.db_conn_load_osm_data import get_osm_ways

import logging
logger = logging.getLogger(__name__)



# scipy is optional dependency for projected nearest-neighbor search
try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None
    

# scikit-learn is optional dependency for unprojected nearest-neighbor search
try:
    from sklearn.neighbors import BallTree
except ImportError:  # pragma: no cover
    BallTree = None


## https://stackoverflow.com/questions/28246425/python-convert-a-list-of-nested-tuples-into-a-dict
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))




def assign_nearest_nodes_to_buildings(graph_subdiv, buildings_w_loads_df):
    
    """
    assign nearest nodes of graph to buildings by euclidean distance.
    """

    X = buildings_w_loads_df['x'].tolist()
    Y = buildings_w_loads_df['y'].tolist()

    buildings_w_loads_df['nn'], buildings_w_loads_df['nn_dist'] = ox.nearest_nodes(
        graph_subdiv, X, Y, return_dist=True)
    buildings_w_loads_df['nn_coords'] = buildings_w_loads_df['nn'].apply(
        lambda row : Point(graph_subdiv.nodes[row]['x'], graph_subdiv.nodes[row]['y']))

    return buildings_w_loads_df


def identify_street_loads(buildings_w_loads_df, graph, get_number_households=False):
    """
    identify street_loads
    street_loads are grouped for lv level only.
    capacity of loads of mv level are not included.
    """

    mv_lv_level_threshold = get_config_osm('mv_lv_threshold_capacity')

    # keep only street_load_nodes and endpoints
    loads = buildings_w_loads_df.copy()

    loads.loc[loads.capacity > mv_lv_level_threshold, 'capacity'] = 0
    street_loads = loads.groupby(['nn']).capacity.sum().reset_index().set_index('nn')

    if get_number_households:
        household_loads = loads.groupby(['nn']).number_households.sum().reset_index().set_index('nn')
        return street_loads, household_loads

    else:

        return street_loads



def get_cluster_graph_and_nodes(simp_graph, labels):
    """
    assign cluster Id to graph nodes
    """

    # assign cluster Id to graph nodes
    node_cluster_dict = dict(zip(list(simp_graph.nodes), labels))
    nx.set_node_attributes(simp_graph, node_cluster_dict, name='cluster')   
    nodes_w_labels = ox.graph_to_gdfs(simp_graph, nodes=True, edges=False)

    return simp_graph, nodes_w_labels


def loads_in_ons_dist_threshold(dm_cluster, cluster_nodes, osmid):
    """
    check max dist between loads and station
    return False if loads_ not in_ons_dist_threshold
    e.g. max dist > 1500m
    """

    ons_dist_threshold = get_config_osm('ons_dist_threshold')
    ons_max_branch_dist = dm_cluster[cluster_nodes.index(osmid)].max()

    if ons_max_branch_dist > ons_dist_threshold:
        return False

    else:
        return True


def get_mvlv_subst_loc_list(cluster_graph, nodes, street_loads_df, labels, n_cluster, check_distance_criterion=True):
    """
    identify position of station at street load center
    get list of location of mvlv substations for load areal
    n_cluster: number of cluster
    """

    mvlv_subst_list = []
    valid_cluster_distance = True

    for i in range(n_cluster):

        df_cluster = nodes[nodes['cluster'] == i]
        cluster_nodes = list(df_cluster.index)

        # create distance matrix for cluster
        cluster_subgraph = cluster_graph.subgraph(cluster_nodes)
        dm_cluster = nx.floyd_warshall_numpy(cluster_subgraph, nodelist=cluster_nodes, weight='length')

        # map cluster_loads with capacity of street_loads
        cluster_loads = pd.Series(cluster_nodes).map(street_loads_df.capacity).fillna(0).tolist()

        # compute location of substation based on load center
        load_vector = np.array(cluster_loads) #weighted
        unweighted_nodes = dm_cluster.dot(load_vector)

        osmid = cluster_nodes[int(np.where(unweighted_nodes == np.amin(unweighted_nodes))[0][0])]

        if check_distance_criterion:
            if not loads_in_ons_dist_threshold(dm_cluster, cluster_nodes, osmid):
                # return empty list and False for cluster validity
                # due to distance threshold to ons is trespassed
                valid_cluster_distance = False
                return mvlv_subst_list, valid_cluster_distance

        mvlv_subst_loc = cluster_graph.nodes[osmid]
        mvlv_subst_loc['osmid'] = osmid
        mvlv_subst_loc['graph_district'] = cluster_subgraph
        mvlv_subst_list.append(mvlv_subst_loc)

    return mvlv_subst_list, valid_cluster_distance


# mv trafo placement
def connect_mv_loads_to_graph(cluster_graph, osm_id_building, row):
    """
    For mv loads, add edge from building to graph instead
    keeping trafo at nearest node of building in graph.
    """
    # set values for mv
    name = osm_id_building
    x = row.raccordement_building.x
    y = row.raccordement_building.y

    # add node to graph
    cluster_graph.add_node(name,x=x,y=y, node_type='non_synthetic', cluster=None)
    # add edge to graph
    line = LineString([row.raccordement_building, row.nn_coords])
    cluster_graph.add_edge(name, row.nn,
                           geometry=line,length=line.length,highway='trafo_graph_connect')
    cluster_graph.add_edge(row.nn, name,
                           geometry=line,length=line.length,highway='trafo_graph_connect')


# logic for filling zeros
def get_lvgd_id(la_id_db, cluster_id, max_n_digits=10):
    """
    param: la_id_db, cluster_id, max_n_digits
           max_n_digits defines number of digits a la or lvgd should have
    due to la ids and lvgd ids should have a length of 10 fill zeros
    new id looks like la_id (what is id_db) + filling zeros + cluster id of lvgd
    return new id
    """
    # getting la_id and its length given by ding0
    la_id = str(la_id_db)

    # getting cluster id and its length computed in new approach
    lvgd_id = str(cluster_id)
    need_to_fill_n_digits = max_n_digits - len(la_id)

    return int(la_id + lvgd_id.zfill(need_to_fill_n_digits))


'''
DEPRECATED DUE TO MV LOADS ARE NOT IN LOAD AREA ANYMORE
def add_mv_load_station_to_mvlv_subst_list(loads_mv_df, mvlv_subst_list, nodes_w_labels):

    """
    add_mv_load_station_to_mvlv_subst_list
    """

    loads_mv_df['x'] = loads_mv_df.apply(lambda x: x.nn_coords.x, axis=1)
    loads_mv_df['y'] = loads_mv_df.apply(lambda x: x.nn_coords.y, axis=1)

    loads_mv_df['node_type'] = loads_mv_df.index.map(nodes_w_labels.node_type)

    loads_mv_df['osmid'] = loads_mv_df['nn']
    loads_mv_df['load_level'] = 'mv'
    loads_mv_df['graph_district'] = None


    mvlv_subst_list += loads_mv_df.to_dict(orient='records')

    return mvlv_subst_list
'''



'''
def simplify_graph(G, street_load_nodes, strict=True, remove_rings=True):
    """
    Simplify a graph's topology by removing interstitial nodes.
    Simplifies graph topology by removing all nodes that are not intersections
    or dead-ends. Create an edge directly between the end points that
    encapsulate them, but retain the geometry of the original edges, saved as
    a new `geometry` attribute on the new edge. Note that only simplified
    edges receive a `geometry` attribute. Some of the resulting consolidated
    edges may comprise multiple OSM ways, and if so, their multiple attribute
    values are stored as a list.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other
        rules but have incident edges with different OSM IDs. Lets you keep
        nodes at elbow two-way intersections, but sometimes individual blocks
        have multiple OSM IDs within them too.
    remove_rings : bool
        if True, remove isolated self-contained rings that have no endpoints
    Returns
    -------
    G : networkx.MultiDiGraph
        topologically simplified graph, with a new `geometry` attribute on
        each simplified edge
    """
    if "simplified" in G.graph and G.graph["simplified"]:  # pragma: no cover
        raise Exception("This graph has already been simplified, cannot simplify it again.")

    #utils.log("Begin topologically simplifying the graph...")
    logger.info("Begin topologically simplifying the graph...")

    # define edge segment attributes to sum upon edge simplification
    attrs_to_sum = {"length", "travel_time"}

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    initial_node_count = len(G)
    initial_edge_count = len(G.edges)
    all_nodes_to_remove = []
    all_edges_to_add = []
    
    
    endpoints = [n for n in G.nodes if ox.simplification._is_endpoint(G, n, strict=True)]
    nodes_to_keep = list(set(endpoints + street_load_nodes))

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, nodes_to_keep, strict=strict):

        # add the interstitial edges we're removing to a list so we can retain
        # their spatial geometry
        path_attributes = dict()
        for u, v in zip(path[:-1], path[1:]):

            # there should rarely be multiple edges between inter_get_paths_to_simplify(G, nodes_to_keep, strict=True):stitial nodes
            # usually happens if OSM has duplicate ways digitized for just one
            # street... we will keep only one of the edges (see below)
            edge_count = G.number_of_edges(u, v)
            if edge_count != 1:
                #utils.log(f"Found {edge_count} edges between {u} and {v} when simplifying")
                logger.info(f"Found {edge_count} edges between {u} and {v} when simplifying")

            # get edge between these nodes: if multiple edges exist between
            # them (see above), we retain only one in the simplified graph
            edge_data = G.edges[u, v, 0]
            
            edge_data.pop('geometry', None) ### NECESSARY FOR THIS PARTIC. CASE (PAUL: if geometry in attributes)
            
            for attr in edge_data:
                if attr in path_attributes:
                    # if this key already exists in the dict, append it to the
                    # value list
                    path_attributes[attr].append(edge_data[attr])
                else:
                    # if this key doesn't already exist, set the value to a list
                    # containing the one value
                    path_attributes[attr] = [edge_data[attr]]

        # consolidate the path's edge segments' attribute values
        for attr in path_attributes:
            if attr in attrs_to_sum:
                # if this attribute must be summed, sum it now
                path_attributes[attr] = sum(path_attributes[attr])
            elif len(set(path_attributes[attr])) == 1:
                # if there's only 1 unique value in this attribute list,
                # consolidate it to the single value (the zero-th):
                path_attributes[attr] = path_attributes[attr][0]
            else:
                # otherwise, if there are multiple values, keep one of each
                path_attributes[attr] = list(set(path_attributes[attr]))

        # construct the new consolidated edge's geometry for this path
        path_attributes["geometry"] = LineString(
            [Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in path]
        )

        # add the nodes and edge to their lists for processing at the end
        all_nodes_to_remove.extend(path[1:-1])
        all_edges_to_add.append(
            {"origin": path[0], "destination": path[-1], "attr_dict": path_attributes}
        )

    # for each edge to add in the list we assembled, create a new edge between
    # the origin and destination
    for edge in all_edges_to_add:
        G.add_edge(edge["origin"], edge["destination"], **edge["attr_dict"])

    # finally remove all the interstitial nodes between the new edges
    G.remove_nodes_from(set(all_nodes_to_remove))

    if remove_rings:
        # remove any connected components that form a self-contained ring
        # without any endpoints
        wccs = nx.weakly_connected_components(G)
        nodes_in_rings = set()
        for wcc in wccs:
            if not any(ox.simplification._is_endpoint(G, n) for n in wcc):
                nodes_in_rings.update(wcc)
        G.remove_nodes_from(nodes_in_rings)

    # mark graph as having been simplified
    G.graph["simplified"] = True

    msg = (
        f"Simplified graph: {initial_node_count} to {len(G)} nodes, "
        f"{initial_edge_count} to {len(G.edges)} edges"
    )
    
    logger.warning(msg)
    return G
'''
