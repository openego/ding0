"""
Routing based on ways from OSM.
"""

import networkx as nx
from geoalchemy2.shape import to_shape

from pyproj import CRS

import geopandas as gpd

import numpy as np

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



def build_graph_from_ways(ways):

    """ 
    Build graph based on osm ways
    Based on this graph, the routing will be implemented.
    """
    
    srid = get_config_osm('srid')
    
    
    # New graph for ways
    routed_graph = nx.MultiGraph()
    routed_graph.graph["crs"] = srid

    # keep coords for each node
    node_coords_dict = {}

    
    # init a graph and set srid.
    routed_graph = nx.MultiGraph()
    routed_graph.graph["crs"] = srid

    
    for w in ways:


        # add nodes. will happen automatically when adding edges.
        # routed_graph_tmp.add_nodes_from(w.nodes)

        # add edges
        ix = 0
        for node in w.nodes[:-1]:

            routed_graph.add_edge(w.nodes[ix], w.nodes[ix+1], length=w.length_segments[ix])

            ix += 1
            
        
        # add coords
        ix = 0
        coords_list = list(to_shape(w.geometry).coords)
        
        for node in w.nodes:

            node_coords_dict[node] = coords_list[ix]
            routed_graph.nodes[node]['X'] = coords_list[ix][0]
            routed_graph.nodes[node]['Y'] = coords_list[ix][1]

            ix += 1
        
        
        
    return routed_graph, node_coords_dict





def nearest_nodes(G, X, Y):
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
        nodes_rad = np.deg2rad(nodes[["Y", "X"]])
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
