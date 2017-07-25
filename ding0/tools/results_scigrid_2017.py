"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from ding0.tools import config as cfg_ding0
from ding0.tools import results

cfg_ding0.load_config('config_db_tables.cfg')
cfg_ding0.load_config('config_calc.cfg')
cfg_ding0.load_config('config_files.cfg')
cfg_ding0.load_config('config_misc.cfg')

base_path = "/home/guido/rli_local/ding0_results/"
ding0 = results.ResultsDing0(base_path)

# concat nd-pickles
# first_mvgd = 5
# last_mvgd = 24
# ding0.concat_nd_pickles(list(range(first_mvgd, last_mvgd + 1)))

# concat csv files of larger ranges
# ranges = [tuple([5,15]), tuple([16,24])]
# ding0.concat_csv_stats_files(ranges)

# create results figures and numbers based of concatenated csv file
concat_csv_file_range = [5, 24]
ding0.read_csv_results(concat_csv_file_range)

# calculate stats
# global_stats = ding0.calculate_global_stats()
mvgd_stats = ding0.calculate_mvgd_stats()
print(mvgd_stats)
ding0.plot_cable_length()
# for key in list(global_stats.keys()):
#     print(key, global_stats[key])