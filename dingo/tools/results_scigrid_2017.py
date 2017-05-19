"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from dingo.tools import config as cfg_dingo
from dingo.tools import results

cfg_dingo.load_config('config_db_tables.cfg')
cfg_dingo.load_config('config_calc.cfg')
cfg_dingo.load_config('config_files.cfg')
cfg_dingo.load_config('config_misc.cfg')

base_path = "/home/guido/rli_local/dingo_results/"
dingo = results.ResultsDingo(base_path)

# concat nd-pickles
# first_mvgd = 5
# last_mvgd = 24
# dingo.concat_nd_pickles(list(range(first_mvgd, last_mvgd + 1)))

# concat csv files of larger ranges
# ranges = [tuple([5,15]), tuple([16,24])]
# dingo.concat_csv_stats_files(ranges)

# create results figures and numbers based of concatenated csv file
concat_csv_file_range = [5, 24]
dingo.read_csv_results(concat_csv_file_range)

# calculate stats
# global_stats = dingo.calculate_global_stats()
mvgd_stats = dingo.calculate_mvgd_stats()
print(mvgd_stats)
dingo.plot_cable_length()
# for key in list(global_stats.keys()):
#     print(key, global_stats[key])