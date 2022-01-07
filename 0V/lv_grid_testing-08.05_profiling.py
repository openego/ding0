'''
via snakeviz profiling of
load graph
lod buildings, amenities and merge
parameteization of buildings

cell [1-6]
'''

#TODO:
#need package snakeviz, eg. conda install snakeviz

#Preparation: Change Path to your use
#Analysis: Try to change/ reduce Depth (to 3)

#e.g.:
#cd C:\Users\Robert\anaconda3\envs\ox\Lib\site-packages\ding0
#activate ox
#python -m cProfile -o lv_grid_testing-09.06_profiling.prof lv_grid_testing-09.06_profiling.py
#snakeviz C:\Users\Robert\anaconda3\envs\ox\Lib\site-packages\ding0\lv_grid_testing-09.06_profiling.prof


from ding0.core import NetworkDing0
from ding0.tools.logger import setup_logger
from ding0.tools.results import save_nd_to_pickle
from ding0.tools.plots import plot_mv_topology
from egoio.tools import db
from sqlalchemy.orm import sessionmaker
import oedialect

from ding0.tools import results # to load pickl file


# create new network
nd = NetworkDing0(name='network')


# set ID of MV grid district
mv_grid_districts = [40] # fn



import osmnx as ox

import networkx as nx 

import pandas as pd
from sqlalchemy import func              
from geoalchemy2.shape import to_shape 


from shapely.geometry import Point, Polygon, LineString


import folium



from config.config_lv_grids_osm import get_config_osm
from config.db_conn_local import create_session_osm 

from grid.lv_grid.routing import build_graph_from_ways, \
get_location_substation_at_pi, subdivide_graph_edges

from grid.lv_grid.parameterization import parameterize_by_load_profiles
from grid.lv_grid.clustering import get_n_cluster, cluster_k_means

from grid.lv_grid.geo import get_Point_from_x_y, get_points_in_load_area, \
get_convex_hull_from_points



# TODO set in config file
ding0_default=False
retain_all=True



engine = db.connection(section='oedb_dialect', readonly=True)
session = sessionmaker(bind=engine)()



# load graph by ding0
graph_subdiv, geo_load_area, buildings_w_loads_df = nd.import_mv_grid_districts(
    session, ding0_default, mv_grid_districts_no=mv_grid_districts)
