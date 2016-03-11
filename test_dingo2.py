from dingo.core import NetworkDingo
from dingo.tools import config as cfg_dingo
from oemof import db

import pandas as pd
################################

cfg_dingo.load_config('config_db_tables')

# get engine for database connection
#conn = db.connection(db_section='ontohub_wdb', cfg_file='~/.dingo/config') # <-- TODO: include custom config file from given path
#conn = db.connection(db_section='ontohub_wdb')

nd = NetworkDingo()

nd.import_mv_stations()
nd.import_lv_regions()
