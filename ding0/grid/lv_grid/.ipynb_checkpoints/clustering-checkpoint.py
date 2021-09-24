from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx

from config.config_lv_grids_osm import get_config_osm 



def get_cluster_numbers(la_peak_loads):
    """
    caculate the number of clusters for load areal based on peak load / avg
    peak loads < 200 kW are accumulated
    add a additional_trafo_capacity to 
    ensure a trafos is loaded to 70% max.
    """
    
    cum_peak_load = la_peak_loads.loc[la_peak_loads.capacity < get_config_osm('mv_lv_threshold_capacity')].capacity.sum()
    
    return int(np.ceil(cum_peak_load / get_config_osm('avg_trafo_size')) * get_config_osm('additional_trafo_capacity'))




def apply_AgglomerativeClustering(G, k, round_decimals=True):
    
    """
    for graph: G apply_AgglomerativeClustering for k cluster
    round_decimals: True makes it reproducible due to coordinates
    may be rounded in different ways, depending on ram/ hardware.
    return labels
    """
    
    if len(G.nodes) > 1:

        X = []    # collect nodes
        for node in G.nodes:
            X.append((G.nodes[node]['x'],G.nodes[node]['y']))
        X = np.array(X)

        adj_mat_sparse = nx.adjacency_matrix(G)


        # ensure number of clusters <= number of buildings 
        if k > len(X):
            k=len(X)


        if round_decimals:

            X = np.round_(X, decimals=4, out=None)

        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=adj_mat_sparse).fit(X)
        
        return clustering.labels_
        
        
    # graph has only one node
    else: return np.array([0])
    
