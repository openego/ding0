from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx
import pandas as pd

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




def apply_AgglomerativeClustering(simp_graph, k, round_decimals=True):
    
    """
    for graph: simp_graph apply_AgglomerativeClustering for k cluster
    round_decimals: True makes it reproducible due to coordinates
    may be rounded in different ways, depending on ram/ hardware.
    return labels
    """

    X = []    # collect nodes
    for node in simp_graph.nodes:
        X.append((simp_graph.nodes[node]['x'],simp_graph.nodes[node]['y']))
    X = np.array(X)

    adj_mat_sparse = nx.adjacency_matrix(simp_graph)


    # ensure number of clusters <= number of buildings 
    if k > len(X):
        k=len(X)
    
    
    if round_decimals:
        
        X = np.round_(X, decimals=4, out=None)

    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=adj_mat_sparse).fit(X)
    


    return clustering.labels_

