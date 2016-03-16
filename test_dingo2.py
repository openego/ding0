from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo
#from dingo.grid.lv_grid import
from oemof import db


cfg_dingo.load_config('config_db_tables')

# get engine for database connection
#conn = db.connection(db_section='ontohub_wdb', cfg_file='~/.dingo/config') # <-- TODO: include custom config file from given path (+input for oemof)
#conn = db.connection(db_section='ontohub_wdb')

nd = NetworkDingo(name='network')

conn = db.connection(db_section='ontohub_oedb')

mv_regions=[1,2]

nd.import_mv_regions(conn, mv_regions)

# cre
# create_lv_stations(network)