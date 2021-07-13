from sklearn.cluster import KMeans

from math import ceil

from config.config_lv_grids_osm import get_config_osm 



def get_n_cluster(buildings_w_loads_df):
    
    """Get n_cluster based on capacity in area. 
       ceil(n_cluster).
       Only loads < 200 kW are considered.
    """
    
    
    mv_lv_threshold_capacity  = get_config_osm('mv_lv_threshold_capacity')
    average_cluster_capacity  = get_config_osm('avg_cluster_capacity')
    additional_trafo_capacity = get_config_osm('additional_trafo_capacity')
    
    capacity_to_cover = buildings_w_loads_df.loc[buildings_w_loads_df['capacity'] < mv_lv_threshold_capacity, 'capacity'].sum()
    capacity_to_cover *= additional_trafo_capacity
    
    return ceil(capacity_to_cover / average_cluster_capacity)




def cluster_k_means(nodes, n_clusters):
    ''' Cluster nodes by x,y as lat, lon
    '''    
    
    kmeans = KMeans(n_clusters=n_clusters)
    
    y = kmeans.fit_predict(nodes[['x', 'y']])

    nodes['Cluster'] = y
    
    
    return nodes