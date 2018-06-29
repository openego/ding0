from ding0.tools.results import export_network, load_nd_from_pickle, export_data_tocsv
import json


grid_id = '76';
base_path = "/home/local/RL-INSTITUT/inga.loeser/ding0/20170922152523/ding0_grids__"
path = ''.join([base_path, grid_id])

# Load network and run export function
nw = load_nd_from_pickle(filename=''.join([path, '.pkl']))
run_id, lv_grid, lv_gen, lv_cd, lv_stations, lv_trafos, lv_loads, mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, mv_trafos, mv_loads, edges, mapping = export_network(nw)

# Create directory for exported data
import os
if not os.path.exists(os.path.join(path, run_id)):
    os.makedirs(os.path.join(path, run_id))

export_data_tocsv(path, run_id, lv_grid, lv_gen, lv_cd, lv_stations, lv_trafos, lv_loads, mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, mv_trafos, mv_loads, edges, mapping )


# Save metadata

metadata = nw.metadata #keys: version, mv_grid_districts, database_tables, data_version, assumptions, run_id
with open(os.path.join(path, run_id, 'Ding0_{}.meta'.format(run_id)),'w') as f: #'Ding0.meta'),'w') as f: #
    json.dump(metadata, f)

print(mapping)
