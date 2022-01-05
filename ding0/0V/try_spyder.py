import warnings

from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


import os
import sys

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


# TODO set in config file
ding0_default=False
retain_all=False #  weil sonst graphen außerhalb des polys unverbunden zum graphen beibehalten werden


engine = db.connection(section='oedb_dialect', readonly=True)
session = sessionmaker(bind=engine)()


print('ding0_default', ding0_default)
if ding0_default:
    
    lv_stations, lv_grid_districts = nd.import_mv_grid_districts(session, 
                                                                 mv_grid_districts_no=mv_grid_districts)

else:
    
    mvlv_subst_list, cluster_graph, nodes_w_labels = nd.import_mv_grid_districts(session, ding0_default,
                                                               mv_grid_districts_no=mv_grid_districts,
                                                               need_parameterization=True,
                                                               retain_all=True,
                                                               truncate_by_edge=False)
    
    

import pandas as pd
import osmnx as ox

nc = ox.plot.get_node_colors_by_attr(cluster_graph, attr='cluster', cmap="Set3") #

fig, ax = ox.plot_graph(cluster_graph, node_color=nc, node_size=50, edge_color='w', edge_linewidth=1, show=False, close=False)

for station in mvlv_subst_list:
    
    ax.scatter(station.get('x'),station.get('y'), color='red')
