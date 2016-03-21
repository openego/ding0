from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo
from oemof import db

import matplotlib.pyplot as plt


plt.close('all')

cfg_dingo.load_config('config_db_tables')

# get engine for database connection
#conn = db.connection(db_section='ontohub_wdb', cfg_file='~/.dingo/config') # <-- TODO: include custom config file from given path (+input for oemof)
#conn = db.connection(db_section='ontohub_wdb')

nd = NetworkDingo(name='network')

conn = db.connection(db_section='ontohub_oedb')
#conn = db.connection(section='ontohub_oedb_remote')

#mv_regions=[106, 125, 500, 722, 887, 1049] # some MV regions from SPF region
mv_regions=[106] # some MV regions from SPF region

nd.import_mv_regions(conn, mv_regions)


# cre
# create_lv_stations(network)

conn.close()

nd._mv_regions[0].mv_grid.routing()
nd._mv_regions[0].mv_grid.graph_draw()