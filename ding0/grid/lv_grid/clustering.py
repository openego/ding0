from sklearn.cluster import AgglomerativeClustering
import numpy as np

from config.config_lv_grids_osm import get_config_osm 



def get_cluster_numbers(la_peak_loads):
    """
    caculate the number of clusters for load areal based on peak load / avg
    peak loads < 200 kW are accumulated
    add a additional_trafo_capacity to 
    ensure a trafos is loaded to 70% max.
    """
    
    cum_peak_load = la_peak_loads.loc[la_peak_loads.capacity < get_config_osm('mv_lv_threshold_capacity')].capacity.sum()
    
    return int(np.ceil(cum_peak_load / get_config_osm('avg_trafo_size'))) * get_config_osm('additional_trafo_capacity')




