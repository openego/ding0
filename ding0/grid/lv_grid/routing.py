"""
Routing based on ways from OSM.
"""

import networkx as nx
from geoalchemy2.shape import to_shape

from pyproj import CRS

import geopandas as gpd

import numpy as np

import osmnx as ox

from shapely.geometry import LineString, Point





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
    
    srid = get_config_osm('srid')

    # keep coords for each node
    # TODO: CHECK IF STILL NECESSARY. WAS USED FOR nx.draw(graph, node_coords_dict)
    #       MAY BE DEPRECATED DUE TO ORIGIN GRAOH IS NOT USED ANYMORE
    #node_coords_dict = {}

    
    # init a graph and set srid.
    graph = nx.MultiGraph()
    graph.graph["crs"] = 'epsg:'+str(srid) 

    
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
        
        
        
        
    # convert undirected graph to digraph
    graph = graph.to_directed()  
    
    # truncate by edge for digraph
    #if truncate_by_edge:
    graph = ox.truncate.truncate_graph_polygon(graph, geo_load_area, retain_all=retain_all, truncate_by_edge=truncate_by_edge)
        
        
        
    return graph




def get_sub_graph_list(graph):
    
    """
    get a list containging subgrpaphs for given graph as param.
    """
    
    
    # TODO: IT IS MANDATORY TO HAVE ONE CONNECTED GRAPH
    #       ENSURE IT IS. IF NOT NEED TO BE IMPLEMENTED
    if not nx.is_weakly_connected(graph):

        # TODO NEED AN IMPLEMENTATION FOR MULTIPLE SUBGRAPHS IF NOT CONNECTED
        print('problem ding0 poly doesnt cover connection(ways) of multiple subgraphs to solve. what can we do?')
        
    
    graph_node_generator = nx.weakly_connected_components(graph)
        
    graph_node_list = list(graph_node_generator)

    sub_graph_list = [] 

    for nodes in graph_node_list:
        
        sub_graph_list.append(graph.subgraph(nodes))
        
    
    return sub_graph_list





def DEPRECATED_nearest_nodes(G, X, Y):
    """
    SRC: https://github.com/gboeing/osmnx/blob/main/osmnx/distance.py#L146    
    
    Find the nearest node to a point or to each of several points.
    If `X` and `Y` are single coordinate values, this will return the nearest
    node to that point. If `X` and `Y` are lists of coordinate values, this
    will return the nearest node to each point.
    If the graph is projected, this uses a k-d tree for euclidean nearest
    neighbor search, which requires that scipy is installed as an optional
    dependency. If it is unprojected, this uses a ball tree for haversine
    nearest neighbor search, which requires that scikit-learn is installed as
    an optional dependency.
    Parameters
    ----------
    G : networkx.MultiDiGraph
        graph in which to find nearest nodes
    X : float or list
        points' x (longitude) coordinates, in same CRS/units as graph and
        containing no nulls
    Y : float or list
        points' y (latitude) coordinates, in same CRS/units as graph and
        containing no nulls
    Returns
    -------
    nn or (nn, dist) : int/list or tuple
        nearest node IDs or optionally a tuple where `dist` contains distances
        between the points and their nearest nodes
    """
    
    
    EARTH_RADIUS_M = srid = get_config_osm('EARTH_RADIUS_M')


    is_scalar = False
    if not (hasattr(X, "__iter__") and hasattr(Y, "__iter__")):
        # make coordinates arrays if user passed non-iterable values
        is_scalar = True
        X = np.array([X])
        Y = np.array([Y])

    if np.isnan(X).any() or np.isnan(Y).any():  # pragma: no cover
        raise ValueError("`X` and `Y` cannot contain nulls")
    
    nodes, data = zip(*G.nodes(data=True))
    nodes = gpd.GeoDataFrame(data, index=nodes)
    
    
    if CRS.from_user_input(G.graph["crs"]).is_projected:
        # if projected, use k-d tree for euclidean nearest-neighbor search
        if cKDTree is None:  # pragma: no cover
            raise ImportError("scipy must be installed to search a projected graph")
        dist, pos = cKDTree(nodes).query(np.array([X, Y]).T, k=1)
        nn = nodes.index[pos]

    else:
        # if unprojected, use ball tree for haversine nearest-neighbor search
        if BallTree is None:  # pragma: no cover
            raise ImportError("scikit-learn must be installed to search an unprojected graph")
        # haversine requires lat, lng coords in radians
        nodes_rad = np.deg2rad(nodes[["y", "x"]])
        points_rad = np.deg2rad(np.array([Y, X]).T)
        dist, pos = BallTree(nodes_rad, metric="haversine").query(points_rad, k=1)
        dist = dist[:, 0] * EARTH_RADIUS_M  # convert radians -> meters
        nn = nodes.index[pos[:, 0]]

    # convert results to correct types for return
    nn = nn.tolist()
    dist = dist.tolist()
    if is_scalar:
        nn = nn[0]
        dist = dist[0]

    return nn, dist




def get_location_substation_at_pi(graph_lv_grid, nodes):
    ''' calculate power distance pi
        pi=dist*capacity
        param  nodes
        return node id with min(pi)
    '''
    
    # init power_distance
    power_distance = 99999999
    update_power_distance=True # check if there is a need to update, init with True
    nodes_to_check = nodes.nn.astype('int64').tolist()
    
    # init with first node
    # in case there is only 1 node in nodes_to_check, its dipower_distance_localst would be 0.
    power_distance_location = nodes_to_check[0]


    for node_to_check in nodes_to_check: # check all node to every other node and calc its distance

        # calc dist and power distance to all other nodes
        # power distance: pi=dist*capacity     
        for checking_node_id, checking_node in nodes.iterrows():
            
            
            dist = nx.shortest_path_length(graph_lv_grid, checking_node.nn, node_to_check, weight='length')
            power_distance_local = dist * checking_node.capacity

            if power_distance_local > 0:

                if power_distance > power_distance_local:
                    
                    update_power_distance=False

                    power_distance = power_distance_local
                    power_distance_location = node_to_check
                    
                 
    # get x,y of pi
    xy = nodes.loc[nodes['nn'] == power_distance_location][['x','y']].iloc[0]
    
    x = xy.x
    y = xy.y
    
    return x,y 



# TODO: CHECK AND DECIDE IF WE WANNA WORK WITH LISTS CONTAINING CUBGRAPHS
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
    
    graph = ox.get_undirected(graph)
    
        
        
    # TODO: ADD meter_sergments to config to be able to set 20m individually. divide edge into 20m segments ###
    #       ADD to_crs='epsg:3035' to cfg
    dist_edge_segments = 20
    concat_0 = "0" # concat edge_id with 0 before inrcrementing synthetic ids to concat


    graph = ox.project_graph(graph, to_crs='epsg:3035')
    

    graph_subdiv = graph.copy()
    edges = graph.edges()

    edge_data = []
    node_data = []

    for u,v in edges:

        linestring = LineString([(graph.nodes[u]['x'],graph.nodes[u]['y']), 
                                 (graph.nodes[v]['x'],graph.nodes[v]['y'])])
        vertices_gen = ox.utils_geo.interpolate_points(linestring, dist_edge_segments) 
        vertices = list(vertices_gen) 
        highway = graph.edges[u,v,0]['highway']
        osmid = graph.edges[u,v,0]['osmid']
        fromid = graph.edges[u,v,0]['from']
        toid = graph.edges[u,v,0]['to']
        edge_id = fromid + toid
        vertex_node_id = []


        for num,node in enumerate(list(vertices)[1:-1], start=1):
            x,y = node[0],node[1] ###
            
            # ensure synthetic node id does not exist in graph.
            # increment name by 1 if exist and check again
            skip_num_by=0
            while True:

                # first name edge_id + 0 + 1(=num) + 0(=skip_num_by)
                name = int(str(edge_id) + concat_0 + str(num+skip_num_by))
                if name in graph.nodes():

                    print('TAKE CARE! THIS SYNTHETIC NODE ALREADY EXISTS IN GRAPH. NODE_ID', name)
                    skip_num_by +=1

                else:

                    break
            
            
            
            node_data.append([name,x,y,(u,v)])
            vertex_node_id.append(name)

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

    for name,x,y,pos in node_data:
        graph_subdiv.add_node(name,x=x,y=y,node_type='synthetic')
        
        
    return graph_subdiv


def assign_nearest_nodes_to_buildings(graph_subdiv, buildings_w_loads_df):
    
    """
    assign nearest nodes of graph to buildings
    """
    
    # TODO: PROJECT TO GET DIST. WHAT ELSE CAN BE DONE?
    graph_subdiv_4326 = ox.project_graph(graph_subdiv, to_crs='epsg:4326')

    X = buildings_w_loads_df['x'].tolist()
    Y = buildings_w_loads_df['y'].tolist()

    buildings_w_loads_df['nn'], buildings_w_loads_df['nn_dist'] = ox.nearest_nodes(
        graph_subdiv_4326, X, Y, return_dist=True)
    buildings_w_loads_df['nn_coords'] = buildings_w_loads_df['nn'].apply(
        lambda row : Point(graph_subdiv_4326.nodes[row]['x'], graph_subdiv_4326.nodes[row]['y']))

    
    return buildings_w_loads_df
