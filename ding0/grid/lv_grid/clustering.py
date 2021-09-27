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




def apply_AgglomerativeClustering(G, inner_node_list, k, round_decimals=True):
    
    """
    for graph: G apply_AgglomerativeClustering for k cluster
    round_decimals: True makes it reproducible due to coordinates
    may be rounded in different ways, depending on ram/ hardware.
    return labels
    """
    
    G_inner = G.subgraph(inner_node_list)
    
    if len(inner_node_list) > 1:

        X = []    # collect nodes
        for node in G_inner.nodes:
            
            X.append((G_inner.nodes[node]['x'],G_inner.nodes[node]['y']))
            
        X = np.array(X)
        

        adj_mat_sparse = nx.adjacency_matrix(G_inner)


        # ensure number of clusters <= number of buildings 
        if len(X) < k:
            k=len(X)


        if round_decimals:

            X = np.round_(X, decimals=4, out=None)

        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward', connectivity=adj_mat_sparse).fit(X)
        
        # return dict containing G_inner.nodes: ClusterId
        graph_cluster_dict = dict(zip(list(G_inner.nodes), clustering.labels_))
            
        
        # set -1 for nodes without a cluster id
        # todo: write -1 in config?
        node_no_cluster_list = list(set(G.nodes)-set(G_inner.nodes))
        graph_no_cluster_dict = dict(zip(node_no_cluster_list, [-1] * len(node_no_cluster_list)))
        
        # merge dict
        graph_cluster_dict = graph_cluster_dict | graph_no_cluster_dict
        
        return graph_cluster_dict
        
        
    # graph has only one node
    # return node:clusterid={node: 0}
    else: return dict(zip(list(G_inner.nodes), [0]))
    
