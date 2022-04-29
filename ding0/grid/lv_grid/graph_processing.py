"""
Graph processing.
"""
from geoalchemy2.shape import to_shape

from config.config_lv_grids_osm import get_config_osm 
import logging
logger = logging.getLogger('ding0')

import pandas as pd
from itertools import combinations

import networkx as nx
import osmnx as ox

import geopandas as gpd
from shapely.geometry import LineString, Point, MultiLineString, Polygon
from shapely.ops import linemerge

# src: https://stackoverflow.com/questions/28246425/python-convert-a-list-of-nested-tuples-into-a-dict
flatten = lambda *n: (e for a in n for e in (flatten(*a) if isinstance(a, (tuple, list)) else (a,)))


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
    graph.graph["source"] = 'osm'

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


def create_simple_synthetic_graph(geo_load_area):
    """
    Build synthetic graph with one node containing
    coordinates of geo_load_area centroid
    """
    # init a graph and set srid and source parameter.
    graph = nx.MultiDiGraph()
    graph.graph["crs"] = 'epsg:' + str(get_config_osm('srid'))
    graph.graph["source"] = 'synthetic'

    # add id and coordinates of area centroid
    node_id = 1
    la_geo_center = geo_load_area.centroid
    node_data = {'x': la_geo_center.x, 'y': la_geo_center.y, 'node_type': 'synthetic'}
    graph.add_node(node_id, **node_data)

    return graph, node_id


def create_buffer_polygons(polygon):

    """
    Create buffer polygons based on origin load area geometry.
    Exterior load area geometry is used.
    """
    # load buffer_distance from config
    buffer_list = get_config_osm('buffer_distance')

    buffer_poly_list = [polygon.convex_hull.buffer(buffer_dist) for buffer_dist in buffer_list] 
    buffer_poly_list.insert(0, Polygon(polygon.exterior).buffer(buffer_list[0]))
    # PAUL NEW # due to fast within operator, import poly needs to be buffered
    
    return buffer_poly_list


def graph_nodes_outside_buffer_polys(graph, ways_sql_df, buffer_poly_list):
    
    """
    src: https://www.matecdev.com/posts/point-in-polygon.html
    For each buffer polygon identify graph nodes outside buffer polygon
    Return nested node list of nodes to remove.
    """

    nodes = flatten(ways_sql_df.nodes.tolist())
    points = flatten(ways_sql_df.geometry.tolist())

    nodes_list = list(nodes)
    points_list = [list(p) for p in list(points)]
    points_list = [Point(item) for sublist in points_list for item in sublist]

    nodes = pd.DataFrame(points_list, index=nodes_list, columns=['geometry'])
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry='geometry', crs=graph.graph["crs"])

    polys_df = pd.DataFrame(buffer_poly_list, index=[i for i in range(len(buffer_poly_list))], columns=['geometry'])
    polys_gdf = gpd.GeoDataFrame(polys_df, geometry='geometry', crs=graph.graph["crs"])

    PointsInPolys = gpd.tools.sjoin(nodes_gdf, polys_gdf, op="within", how='left')
    
    nodes_to_remove = [list(set(nodes_list) - set(PointsInPolys.loc[PointsInPolys['index_right'] == poly].index.tolist())) for poly in range(len(buffer_poly_list))]

    return nodes_to_remove


def truncate_graph_nodes(graph, nested_node_list, poly_idx):
    """
    Remove nodes from buffer graphs based on nested node list
    and index of buffer polygon
    """
    G = graph.copy()
    G.remove_nodes_from(nested_node_list[poly_idx])
    return G


def compose_graph(outer_graph, graph_subdiv):

    """
    composed_graph has all nodes & edges of both graphs, including attributes
    Where the attributes conflict, it uses the attributes of graph_subdiv_directed.
    compose conn_graph with graph_subdiv_directed to have subdivides edges of
    graph_subdiv_directed in conn_graph
    """
    composed_graph = nx.compose(outer_graph, graph_subdiv)

    if not nx.is_weakly_connected(composed_graph):
        logger.warning('composed_graph not connected')
        composed_graph = ox.utils_graph.get_largest_component(composed_graph)

    return composed_graph


def nodes_connected_component(G, inner_node_list):
    """
    Return largest connected component of origin graph based on passed node list.
    """
    nodes = [(cc, len(set(cc)&set(inner_node_list))) for cc in nx.weakly_connected_components(G)]
    largest_cc = max(nodes,key=lambda item:item[1])[0]
    G = nx.MultiDiGraph(G.subgraph(largest_cc))
    return G


def get_fully_conn_graph(G, nlist): #nested_node_list
    """
    Return largest connected component of buffered graph that contains all nodes from nodes to connect
    Iterating over all buffered polygon graphs
    """

    poly_idx = 0 
    G_min = truncate_graph_nodes(G, nlist, poly_idx)
    max_it = len(nlist) - 1

    if nx.number_weakly_connected_components(G_min) > 1:

        G_c_max = nodes_connected_component(G, G_min.nodes())
        nodes_to_connect = set(G_c_max.nodes) & set(G_min.nodes)
        G_c = nodes_connected_component(G_min, G_min.nodes())

        while not all(node in G_c.nodes for node in nodes_to_connect):

            poly_idx += 1
            G_buff = truncate_graph_nodes(G, nlist, poly_idx)
            G_c = nodes_connected_component(G_buff, nodes_to_connect) 

            unconn_nodes = list(set(nodes_to_connect)-set(G_c.nodes))

            if len(unconn_nodes) == 0:

                logger.warning(f'Finding connected graph, iteration {poly_idx} of max. {max_it}.')
                break

            if poly_idx >= max_it:

                logger.warning(f'Finding connected graph, max. iterations {max_it} trespassed. Break.')
                break

        else:
            logger.warning(f'Found connected graph, iteration {poly_idx} of max. {max_it}.')
            return G_c

    else:

        logger.warning(f'Graph already fully connected.')
        G_c = G_min

    return G_c

###

def split_conn_graph(conn_graph, inner_node_list):
    """
    Divide graoh in inner and outer components.
    based on plygon borders.
    """
    
    inner_graph = conn_graph.subgraph(inner_node_list).copy()
    inner_edges = inner_graph.edges()

    outer_graph = conn_graph
    outer_graph.remove_edges_from(inner_edges)
    outer_graph.remove_nodes_from(list(nx.isolates(outer_graph)))
    
    return(inner_graph, outer_graph)


def get_outer_conn_graph(G, inner_node_list):
    """
    Identify components of outter graph suitable for connecting
    inner graph components.
    """

    common_nodes = set(G.nodes()) & set(inner_node_list)
    G = simplify_graph_adv(G, list(common_nodes))
    G = remove_unloaded_deadends(G, list(common_nodes))
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


def add_edge_geometry_entry(G):
    """
    todo: check if to remove if not then write doc.
    looks like deprecated?
    """
    edges_wo_geometry = [(u,v,k,d) for u,v,k,d in G.edges(keys=True, data=True) if 'geometry' not in d]
    for u,v,k,d in edges_wo_geometry:
        G.edges[u,v,k]['geometry'] = LineString([Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in [u,v]])
        
    return G


#### modified osmnx functions

# modified osmnx function (extended by nodes_to_keep)
# feststellen welche knoten behalten und welche weg. build paths.
def _get_paths_to_simplify(G, nodes_to_keep, strict=True):
    """
    # modified osmnx function (extended by nodes_to_keep)
    street_load_nodes are treated as endpoints and are to keep.

    Generate all the paths to be simplified between endpoint nodes.
    The path is ordered from the first endpoint, through the interstitial nodes,
    to the second endpoint.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        input graph
    strict : bool
        if False, allow nodes to be end points even if they fail all other rules
        but have edges with different OSM IDs
    Yields
    ------
    path_to_simplify : list
    """    
    
    # for each endpoint node, look at each of its successor nodes
    for endpoint in nodes_to_keep:
        for successor in G.successors(endpoint):
            if successor not in nodes_to_keep:
                # if endpoint node's successor is not an endpoint, build path
                # from the endpoint node, through the successor, and on to the
                # next endpoint node
                yield ox.simplification._build_path(G, endpoint, successor, nodes_to_keep)


def simplify_graph_adv(G, street_load_nodes, strict=True, remove_rings=True):
    """
    # modified osmnx function (extended by street_load_nodes=nodes_to_keep)
    street_load_nodes are treated as endpoints and are to keep.
    fct is adapted and graph can be simplified several times.

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
    ### PAUL
    #if "simplified" in G.graph and G.graph["simplified"]:  # pragma: no cover
        #raise Exception("This graph has already been simplified, cannot simplify it again.")

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
    
    if len(endpoints) < 1:
        return G # TODO: check how to simplify without endpoints

    # generate each path that needs to be simplified
    for path in _get_paths_to_simplify(G, nodes_to_keep, strict=strict):
        
        if "simplified" in G.graph and G.graph["simplified"]: ### PAUL
            
            edge_path = [(path[i], path[i+1], next(iter(G.get_edge_data(path[i],path[i+1]).keys()))) for i in range(len(path)-1)]
            linestring = linemerge(MultiLineString([G.edges[edge]['geometry'] for edge in edge_path]))

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
            
            edge_data = G.edges[u, v, next(iter(G.get_edge_data(u,v).keys()))] ### PAUL
            #edge_data = G.edges[u, v, 0]
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
            path_attributes[attr] = list(flatten(path_attributes[attr])) ### PAUL
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
        

        if "simplified" in G.graph and G.graph["simplified"]:
            path_attributes["geometry"] = linestring

        else:
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

def flatten_graph_components_to_lines(G, inner_node_list):
    """
    Build single edges based on outter graph component to connect
    isolated components/ nodes in inner graph/ not buffered area.
    """
    # TODO: add edge tags 'highway' and 'osmid' to shortest path edge

    components = list(nx.weakly_connected_components(G))
    sp_path = lambda p1, p2: nx.shortest_path(G, p1, p2, weight='length') if nx.has_path(G, p1, p2) else None

    nodes_to_remove = []
    edges_to_add = []
    common_nodes = set(G.nodes()) & set(inner_node_list)

    for comp in components:
        conn_nodes = list(comp & common_nodes)
        if len(comp) > 2 and len(conn_nodes) == 1:
            G.remove_nodes_from(comp) # removes unwanted islands / loops

        else: 
            endpoints = combinations(conn_nodes, 2)
            paths = [sp_path(n[0], n[1]) for n in endpoints]
            for path in paths:
                geoms = []
                for u, v in zip(path[:-1], path[1:]):
                    geom = G.edges[u,v,0]['geometry']
                    # deprecated due to add_edge_geometry_entry(G)
                    # try: geom = G.edges[u,v,0]['geometry']
                    # except: geom = LineString([Point((G.nodes[node]["x"], G.nodes[node]["y"])) for node in [u,v]])
                    geoms.append(geom)

                merged_line = linemerge(MultiLineString(geoms))
                edges_to_add.append([path[0], path[-1], merged_line])
                nodes_to_remove.append(list(set(comp) - set(conn_nodes)))

    for nodes in nodes_to_remove:
        G.remove_nodes_from(nodes)

    for edge in edges_to_add:
        G.add_edge(edge[0], edge[1], 0, geometry=edge[2], length=edge[2].length)
        G.add_edge(edge[1], edge[0], 0, geometry=edge[2], length=edge[2].length)

    return G

def remove_detours(G):
    """
    Identify edges with length > 1500m and
    Remove em if path dist > 3 * euclid distance
    """
    edges = [(u,v,k,d['length'], LineString(d['geometry'].boundary).length) for u,v,k,d in G.edges(keys=True, data=True) if d['length'] >= get_config_osm('ons_dist_threshold')]
    for u,v,k,path_dist,euc_dist in edges:
        if path_dist > 3 * euc_dist:
            G.remove_edge(u,v,k)
    return G


def remove_unloaded_deadends(G, nodes_of_interest):
    """
    remove_unloaded_deadends
    nodes_of_interest are mandatory nodes to keep
    """
    while True:
        dead_end_nodes = [node for node,degree in dict(G.degree()).items() if degree == 2]
        dead_end_nodes_to_remove = list(set(dead_end_nodes) - set(nodes_of_interest))
        if len(dead_end_nodes_to_remove) > 0:
            G.remove_nodes_from(dead_end_nodes_to_remove)
            #logger.warning(len(G.nodes))
        else:
            break 
    return G

def remove_parallels(G):
    """
    remove the parallel edge with greater "weight" attribute value
    deprecated ? check remove_parallels_and_loops.
    src: https://github.com/gboeing/osmnx/blob/main/osmnx/utils_graph.py#L341
    """
    # from https://github.com/gboeing/osmnx/blob/main/osmnx/utils_graph.py#L341
    #G = G.copy()
    to_remove = []

    # identify all the parallel edges in the MultiDiGraph
    parallels = ((u, v) for u, v, k in G.edges(keys=True) if k > 0)

    # remove the parallel edge with greater "weight" attribute value
    for u, v in set(parallels):
        k, _ = max(G.get_edge_data(u, v).items(), key=lambda x: x[1]['length'])
        to_remove.append((u, v, k))

    G.remove_edges_from(to_remove)
    return G

def remove_parallels_and_loops(G):
    """
    remove the parallel edge with greater "weight" attribute value
    remove selfloop_edges
    src: https://github.com/gboeing/osmnx/blob/main/osmnx/utils_graph.py#L341
    """
    # from https://github.com/gboeing/osmnx/blob/main/osmnx/utils_graph.py#L341
    #G = G.copy()
    to_remove = []

    # identify all the parallel edges in the MultiDiGraph
    parallels = ((u, v) for u, v, k in G.edges(keys=True) if k > 0)

    # remove the parallel edge with greater "weight" attribute value
    for u, v in set(parallels):
        k, _ = max(G.get_edge_data(u, v).items(), key=lambda x: x[1]['length'])
        to_remove.append((u, v, k))

    G.remove_edges_from(to_remove)
    G.remove_edges_from(nx.selfloop_edges(G))
    
    return G


def remove_unloaded_loops(G, nodes_of_interest):
    """
    remove_unloaded_loops and remove corresponding node to loops
    nodes_of_interest are mandatory nodes to keep
    """
    #G = G.copy()
    simple_node_loops = list(set([edge[0] for edge in nx.selfloop_edges(G) if len(G[edge[0]]) <= 2])) # set due to digraph
    node_loops_to_remove = list(set(simple_node_loops) - set(nodes_of_interest))
    G.remove_edges_from(nx.selfloop_edges(G))
    G.remove_nodes_from(node_loops_to_remove)
    return G


def subdivide_graph_edges(inner_graph): #(inner_graph, inner_node_list):

    """
    subdivide_graph_edges by defined distance value get_config_osm('dist_edge_segments')
    and build new graph considering subdivided edges with additional synthetic nodes.
    TODO: keep information about edge name
          ensure edge name does not exist when adding
    """ 

    graph_subdiv = inner_graph.copy()
    edges = inner_graph.edges()
    edge_data = []
    node_data = []
    unique_syn_nodes = []
    origin_vertices = []

    for u,v in edges:

        origin_vertices.append((v,u))

        if (u,v) in origin_vertices:
            pass
        else:

            linestring = LineString([(inner_graph.nodes[u]['x'],inner_graph.nodes[u]['y']), 
                                     (inner_graph.nodes[v]['x'],inner_graph.nodes[v]['y'])])
            vertices_gen = ox.utils_geo.interpolate_points(linestring, get_config_osm('dist_edge_segments')) #### config
            vertices = list(vertices_gen) 
            highway = inner_graph.edges[u,v,0]['highway']
            osmid = inner_graph.edges[u,v,0]['osmid']
            # fromid = graph.edges[u,v,0]['from'] ### PAUL
            # toid = graph.edges[u,v,0]['to'] ### PAUL
            edge_id = u + v
            vertex_node_id = []

            for num,node in enumerate(list(vertices)[1:-1], start=1):
                x,y = node[0],node[1] ###

                # ensure synthetic node id does not exist in graph.
                # increment name by 1 if exist and check again
                skip_num_by=0
                while True:

                    # first name edge_id + 0 + 1(=num) + 0(=skip_num_by)
                    concat_0 = 0 # todo: check concat_0 is to change. check if node name needs to be str
                    name = int(str(edge_id) + str(concat_0) + str(num+skip_num_by))
                    if name in unique_syn_nodes: #graph_subdiv.nodes():

                        skip_num_by +=1
                        name = int(str(edge_id) + str(concat_0) + str(num+skip_num_by)) ####

                    else:

                        break

                node_data.append([name,x,y,(u,v)])
                vertex_node_id.append(name)
                unique_syn_nodes.append(name)

            if vertices[0] == (inner_graph.nodes[v]['x'],inner_graph.nodes[v]['y']): ###
                vertex_node_id.insert(0, v)
                vertex_node_id.append(u)
            else:
                vertex_node_id.insert(0, u)
                vertex_node_id.append(v)

            for i,j in zip(range(len(list(vertices))-1), range(len(vertex_node_id)-1)):
                line = LineString([vertices[i], vertices[i+1]])
                edge_data.append([vertex_node_id[j], vertex_node_id[j+1], line, line.length, highway, osmid])

    #build new graph 
    graph_subdiv.remove_edges_from(edges)

    for u,v,line,leng,highway,osmid in edge_data:
        graph_subdiv.add_edge(u,v,geometry=line,length=leng,highway=highway,osmid=osmid)
        graph_subdiv.add_edge(v,u,geometry=line,length=leng,highway=highway,osmid=osmid)

    for name,x,y,pos in node_data:
        graph_subdiv.add_node(name,x=x,y=y,node_type='synthetic')

    return graph_subdiv #graph_subdiv, edges