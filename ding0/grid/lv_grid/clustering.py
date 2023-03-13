from sklearn.cluster import AgglomerativeClustering
import numpy as np
import networkx as nx

from ding0.config.config_lv_grids_osm import get_config_osm
from ding0.grid.lv_grid.routing import get_mvlv_subst_loc_list, get_cluster_graph_and_nodes

import logging
logger = logging.getLogger(__name__)


def get_cluster_numbers(la_peak_loads, simp_graph):
    """
    caculate the number of clusters for load areal based on peak load / avg
    peak loads < 200 kW are accumulated
    add additional_trafo_capacity to ensure a trafos is loaded to 80% max.
    trafo_loading = 500 / 630
    """
    cum_peak_load = la_peak_loads.loc[la_peak_loads.capacity < get_config_osm('mv_lv_threshold_capacity')].capacity.sum()

    n_cluster = int(np.ceil(cum_peak_load / get_config_osm('avg_trafo_size')) * get_config_osm('additional_trafo_capacity'))
    # workaround if peak load of low voltage loads is zero,
    # TODO: check if load area should be removed -> cfg_ding0.get('mv_routing', 'load_area_threshold')
    if n_cluster == 0:
        n_cluster = 1

    elif n_cluster > len(simp_graph):
        n_cluster = len(simp_graph)

    return n_cluster


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
    cluster_increment_counter_threshold = get_config_osm('cluster_increment_counter_threshold')

    for i in range(len(simp_graph.nodes)):

        # increment n_cluster. n_cluster += 1
        labels = apply_AgglomerativeClustering(simp_graph, n_cluster)

        cluster_graph, nodes_w_labels = get_cluster_graph_and_nodes(simp_graph, labels)

        if cluster_increment_counter > cluster_increment_counter_threshold:
            check_distance_criterion = False

            message = f"cluster_increment_counter > {cluster_increment_counter_threshold} -> " \
                      f"{get_config_osm('ons_dist_threshold')}m distance is not ensured. " \
                      f"Check export: MV {mv_grid_district}, LA {id_db} does not ensure " \
                      f"max dist of {get_config_osm('ons_dist_threshold')}m between station and loads."

            mv_grid_district.network.message.append(message)
            logger.warning(message)

            clustering_successfully = True

        # locate stations for districts
        mvlv_subst_list, valid_cluster_distance = get_mvlv_subst_loc_list(cluster_graph, nodes_w_labels, \
                                                                          street_loads_df, labels, n_cluster, \
                                                                          check_distance_criterion)

        if valid_cluster_distance:

            logger.debug('All clusters are in range.')

            clustering_successfully = True
            break

        else:

            if cluster_increment_counter <= cluster_increment_counter_threshold:

                cluster_increment_counter += 1
                n_cluster += 1
                logger.debug(f'At least one node trespasses dist to substation, for n_clusters = {n_cluster}. '
                             f'Cluster again with n_clusters+=1')

    return clustering_successfully, cluster_graph, mvlv_subst_list, nodes_w_labels
