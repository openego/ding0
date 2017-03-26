from dingo.tools import config as cfg_dingo
from dingo.tools import results

cfg_dingo.load_config('config_db_tables.cfg')
cfg_dingo.load_config('config_calc.cfg')
cfg_dingo.load_config('config_files.cfg')
cfg_dingo.load_config('config_misc.cfg')

base_path = "/home/guido/rli_local/dingo_results/"

first_mvgd = 5
last_mvgd = 24

# read data from multiple files a merge to one
# mv_grid_districts = list(range(first_mvgd,last_mvgd + 1))
# dingo_results = results.ResultsDingo(mv_grid_districts=mv_grid_districts,
#                                      base_path=base_path)
# dingo_results.save_merge_data()

# evaluate dingo grids
dingo_results = results.ResultsDingo(
    filenames={
        'nd': 'dingo_grids_{0}-{1}.pkl'.format(first_mvgd, last_mvgd),
        'nodes': 'mvgd_nodes_stats_{0}-{1}.csv'.format(first_mvgd, last_mvgd),
        'edges': 'mvgd_edges_stats_{0}-{1}.csv'.format(first_mvgd, last_mvgd)
    },
    base_path=base_path)
global_stats = dingo_results.calculate_global_stats()
mvgd_stats = dingo_results.calculate_mvgd_stats()
dingo_results.plot_cable_length()
for key in list(global_stats.keys()):
    print(key, global_stats[key])