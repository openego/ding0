"""
Routing based on ways from OSM.
"""

import osmnx as ox
import networkx as nx

from geoalchemy2.shape import to_shape

from pyproj import CRS

import geopandas as gpd

import numpy as np

from shapely.geometry import LineString, Point

import pandas as pd





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


from config.config_lv_grids_osm import get_config_osm 



def build_graph_from_ways(ways, geo_load_area, retain_all, truncate_by_edge):

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

            #node_coords_dict[node] = coords_list[ix]
            graph.nodes[node]['x'] = coords_list[ix][0]
            graph.nodes[node]['y'] = coords_list[ix][1]
            graph.nodes[node]['node_type'] = 'non_synthetic'

            ix += 1
        
        
        
        
    # convert undirected graph to digraph due to truncate_graph_polygon
    graph = graph.to_directed()  
    
    # truncate by edge for digraph
    if truncate_by_edge:
        
        # todo qcheck if 1000 from config is best solution or calc e.g. with EARTH_RADIUS_M
        graph = ox.truncate.truncate_graph_polygon(graph, geo_load_area, 
                                                   retain_all=False, 
                                                   truncate_by_edge=True, 
                                                   quadrat_width=get_config_osm('quadrat_width'))
        
        # default with lon lat. deprecated due to srid 3035
        #graph = ox.truncate.truncate_graph_polygon(graph, geo_load_area, retain_all=retain_all, truncate_by_edge=truncate_by_edge)
        
        
        
    return graph




def get_sub_graph_list(graph):
    
    """
    get a list containging connected subgrpaphs for given directed graph as param.
    """
    
    # TODO  I: IT IS MANDATORY TO HAVE ONE CONNECTED GRAPH
    #          ENSURE IT IS. IF NOT NEED TO BE IMPLEMENTED
    if not nx.is_weakly_connected(graph):

        # TODO NEED AN IMPLEMENTATION FOR MULTIPLE SUBGRAPHS IF NOT CONNECTED
        print('problem ding0 poly doesnt cover connection(ways) of multiple subgraphs to solve. what can we do?')
        
    
    graph_node_generator = nx.weakly_connected_components(graph)
        
    graph_node_list = list(graph_node_generator)

    sub_graph_list = [] 

    for nodes in graph_node_list:
        
        sub_graph_list.append(graph.subgraph(nodes))
        
    
    return sub_graph_list




# TODO: CHECK AND DECIDE IF WE WANNA WORK WITH LISTS CONTAINING SUBGRAPHS
def apply_subdivide_graph_edges_for_each_subgraph(sub_graph_list):
    
    """
    for each subgfraoh in list subdivide graph
    """
    
    sub_graph_list_subdivided = []
    
    for graph in sub_graph_list:
        
        sub_graph_list_subdivided.append(subdivide_graph_edges(graph))
        
        
    return sub_graph_list_subdivided




def subdivide_graph_edges(graph):
    
    """
    subdivide_graph_edges
    TODO: keep information about edge name
          ensure edge name does not exist when adding
    """    
    
    graph_subdiv = graph.copy()
    edges = graph.edges()

    edge_data = []
    node_data = []
    unique_syn_nodes = []
    origin_vertices = []

    for u,v in edges:

        origin_vertices.append((v,u))

        if (u,v) in origin_vertices:

            pass

        else:

            linestring = LineString([(graph.nodes[u]['x'],graph.nodes[u]['y']), 
                                     (graph.nodes[v]['x'],graph.nodes[v]['y'])])
            vertices_gen = ox.utils_geo.interpolate_points(linestring, get_config_osm('dist_edge_segments')) #### config
            vertices = list(vertices_gen) 
            highway = graph.edges[u,v,0]['highway']
            osmid = graph.edges[u,v,0]['osmid']
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

                        #print('TAKE CARE! THIS SYNTHETIC NODE ALREADY EXISTS IN GRAPH. NODE_ID', name)
                        skip_num_by +=1
                        name = int(str(edge_id) + str(concat_0) + str(num+skip_num_by)) ####

                    else:

                        break

                node_data.append([name,x,y,(u,v)])
                vertex_node_id.append(name)
                unique_syn_nodes.append(name)

            if vertices[0] == (graph.nodes[v]['x'],graph.nodes[v]['y']): ###
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
        # TODO: add lon lat
        graph_subdiv.add_node(name,x=x,y=y,node_type='synthetic')
        
        
    return graph_subdiv




def assign_nearest_nodes_to_buildings(graph_subdiv, buildings_w_loads_df):
    
    """
    assign nearest nodes of graph to buildings
    """

    X = buildings_w_loads_df['x'].tolist()
    Y = buildings_w_loads_df['y'].tolist()

    buildings_w_loads_df['nn'], buildings_w_loads_df['nn_dist'] = ox.nearest_nodes(
        graph_subdiv, X, Y, return_dist=True)
    buildings_w_loads_df['nn_coords'] = buildings_w_loads_df['nn'].apply(
        lambda row : Point(graph_subdiv.nodes[row]['x'], graph_subdiv.nodes[row]['y']))

    
    return buildings_w_loads_df




def identify_nodes_to_keep(buildings_w_loads_df, graph):
    """
    identify_nodes_to_keep
    street_loads are grouped for lv level only.
    capacity of loads of mv level are not included. 
    """
    
    mv_lv_level_threshold = get_config_osm('mv_lv_threshold_capacity')
    
    # keep only street_load_nodes and endpoints
    street_loads = buildings_w_loads_df.copy()
    street_loads.loc[street_loads.capacity > mv_lv_level_threshold, 'capacity'] = 0 # todo: check if working
    street_loads = street_loads.groupby(['nn']).capacity.sum().reset_index().set_index('nn')
    # before
    #street_loads = buildings_w_loads_df.loc[buildings_w_loads_df.capacity < mv_lv_level_threshold].groupby(
    #    ['nn']).capacity.sum().reset_index().set_index('nn')
    street_load_nodes = street_loads.index.tolist()

    endpoints = [n for n in graph.nodes if ox.simplification._is_endpoint(graph, n, strict=True)]
    nodes_to_keep = list(set(endpoints + street_load_nodes))
    
    return nodes_to_keep, street_loads



# modified osmnx function (extended by nodes_to_keep)
# feststellen welche knoten behalten und welche weg. build paths.
def _get_paths_to_simplify(G, nodes_to_keep, strict=True):
    """
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
                
                
                
# modified osmnx function (extended by nodes_to_keep)
# del paths from above and build new graph
def simplify_graph(G, nodes_to_keep, strict=True, remove_rings=True):
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
    print("Begin topologically simplifying the graph...")

    # define edge segment attributes to sum upon edge simplification
    attrs_to_sum = {"length", "travel_time"}

    # make a copy to not mutate original graph object caller passed in
    G = G.copy()
    initial_node_count = len(G)
    initial_edge_count = len(G.edges)
    all_nodes_to_remove = []
    all_edges_to_add = []

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
                print(f"Found {edge_count} edges between {u} and {v} when simplifying")

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
    
    #utils.log(msg)
    print(msg)
    return G



def get_cluster_graph_and_nodes(simp_graph, labels):
    
    """
    get_cluster_graph_and_nodes
    """

    # assign cluster number to nodes V2
    node_cluster_dict = dict(zip(list(simp_graph.nodes), labels))
    nx.set_node_attributes(simp_graph, node_cluster_dict, name='cluster')   
    nodes = ox.graph_to_gdfs(simp_graph, nodes=True, edges=False)

    return simp_graph, nodes




def get_mvlv_subst_loc_list(cluster_graph, nodes, street_loads, labels, n_cluster):
    
    """
    get list of location of mvlv substations for load areal
    only for lv level: building loads < 200 kW (threshold)
    n_cluster: number of cluster
    """

    #nc = ox.plot.get_node_colors_by_attr(cluster_graph, attr='cluster')
    #fig, ax = ox.plot_graph(cluster_graph, node_color=nc, node_size=50, edge_color='w', edge_linewidth=1, show=False, close=False)

    mvlv_subst_list = []
    
    valid_cluster_distance = True
    
    for i in range(n_cluster):
        df_cluster = nodes[nodes['cluster'] == i]
        cluster_nodes = list(df_cluster.index)

        # map cluster_loads with capacity of street_loads
        cluster_loads = pd.Series(cluster_nodes).map(street_loads.capacity).fillna(0).tolist()

        # create distance matrix for cluster
        cluster_subgraph = cluster_graph.subgraph(cluster_nodes)
        dm_cluster = nx.floyd_warshall_numpy(cluster_subgraph, nodelist=cluster_nodes, weight='length')

        # compute location of substation based on load center
        load_vector = np.array(cluster_loads) #weighted
        unweighted_nodes = dm_cluster.dot(load_vector)
        
        osmid = cluster_nodes[int(np.where(unweighted_nodes == np.amin(unweighted_nodes))[0][0])]
        
        
        # todo organize functions
        def loads_in_ons_dist_threshold(osmid):
            
            """
            return False if loads_ not in_ons_dist_threshold
            """
            
            ons_dist_threshold = get_config_osm('ons_dist_threshold')
            
            ons_max_branch_dist = dm_cluster[cluster_nodes.index(osmid)].max()
            
            if ons_max_branch_dist > ons_dist_threshold:
            
                return False
            
            else: return True
            
        
        if not loads_in_ons_dist_threshold(osmid):
            
            # return empty list and False for cluster validity
            # due to distance threshold to ons is trespassed
            return []
        
        
        mvlv_subst_loc = cluster_graph.nodes[osmid]
        mvlv_subst_loc['osmid'] = osmid
        mvlv_subst_loc['load_level'] = 'lv'
        mvlv_subst_loc['graph_district'] = cluster_subgraph
        mvlv_subst_list.append(mvlv_subst_loc)
        
    return mvlv_subst_list



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
