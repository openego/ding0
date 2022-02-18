from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx

from config.config_lv_grids_osm import get_config_osm 
from ding0.grid.lv_grid.routing import get_mvlv_subst_loc_list, get_cluster_graph_and_nodes

import logging
logger = logging.getLogger('ding0')


def get_cluster_numbers(la_peak_loads):
    """
    caculate the number of clusters for load areal based on peak load / avg
    peak loads < 200 kW are accumulated
    add additional_trafo_capacity to ensure a trafos is loaded to 80% max.
    trafo_loading = 500 / 630
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


def distance_restricted_clustering(simp_graph, n_cluster, street_loads_df, mv_grid_district, id_db):
    """
    Apply ward hierarchical AgglomerativeClustering with connectivity constraints for underlying graph
    https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
    Linkage criteria: ward. Connectivity constraints: adjacent matrix from graph
    Locate stations in fct: get_mvlv_subst_loc_list()
    return True if clustering_successfully else False
    """
    clustering_successfully = False  # init False
    cluster_increment_counter = 0  # init 0
    check_distance_criterion = True

    for i in range(len(simp_graph.nodes)):

        # increment n_cluster. n_cluster += 1
        labels = apply_AgglomerativeClustering(simp_graph, n_cluster)

        cluster_graph, nodes_w_labels = get_cluster_graph_and_nodes(simp_graph, labels)

        if cluster_increment_counter > 10:
            check_distance_criterion = False

            logger.warning('cluster_increment_counter > 10. \
            break increment n_cluster. 1500 m distance is not ensured. \
            check export: no_1500m_distance_ensured.txt')

            # write mv and la id to file
            f = open("no_1500m_distance_ensured.txt", "a")
            f.write(f"MV {mv_grid_district}, LA {id_db}\n does not ensure"
                    " max dist of 1500m between station and loads \n")
            f.close()
            clustering_successfully = True

        # locate stations for districts
        mvlv_subst_list, valid_cluster_distance = get_mvlv_subst_loc_list(cluster_graph, nodes_w_labels, \
                                                                          street_loads_df, labels, n_cluster, \
                                                                          check_distance_criterion)

        if valid_cluster_distance:

            logger.warning('all clusters are in range')

            clustering_successfully = True
            break

        else:

            if cluster_increment_counter <= 10:  # todo: write 10 to config

                cluster_increment_counter += 1
                n_cluster += 1
                logger.warning('at least one node trespasses dist to substation. \
                               cluster again with n_clusters+=1')
                logger.warning(f'after increment; n_cluster {n_cluster}')

    return clustering_successfully, cluster_graph, mvlv_subst_list, nodes_w_labels
