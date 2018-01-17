from ding0.tools.results import export_network, load_nd_from_pickle
import json


grid_id = '76';
base_path = "/home/local/RL-INSTITUT/inga.loeser/ding0/20170922152523/ding0_grids__"
path = ''.join([base_path, grid_id])

# Load network and run export function
nw = load_nd_from_pickle(filename=''.join([path, '.pkl']))
test, run_id, lv_gen, lv_cd, lv_stations, lv_trafos, lv_loads, mv_gen, mv_cb, mv_cd, mv_stations, areacenter, mv_trafos, mv_loads, edges, mapping = export_network(nw)

# Create directory for exported data
import os
if not os.path.exists(os.path.join(path, run_id)):
    os.makedirs(os.path.join(path, run_id))

# Export data to csv
def export_network_tocsv(path, table, tablename):
    return table.to_csv(''.join([path, '/', run_id, '/', tablename, '.csv']), ';')

export_network_tocsv(path, lv_gen, 'lv_gen')
export_network_tocsv(path, lv_cd, 'lv_cd')
export_network_tocsv(path, lv_stations, 'lv_stations')
export_network_tocsv(path, lv_trafos, 'lv_trafos')
export_network_tocsv(path, lv_loads, 'lv_loads')
export_network_tocsv(path, mv_gen, 'mv_gen')
export_network_tocsv(path, mv_cd, 'mv_cd')
export_network_tocsv(path, mv_stations, 'mv_stations')
export_network_tocsv(path, mv_trafos, 'mv_trafos')
export_network_tocsv(path, mv_cb, 'mv_cb')
export_network_tocsv(path, mv_loads, 'mv_loads')
export_network_tocsv(path, edges, 'edges')
export_network_tocsv(path, mapping, 'mapping')
export_network_tocsv(path, areacenter, 'areacenter')

# Save metadata

metadata = nw.metadata #keys: version, mv_grid_districts, database_tables, data_version, assumptions, run_id
with open(os.path.join(path, run_id, 'Ding0_{}.meta'.format(run_id)),'w') as f: #'Ding0.meta'),'w') as f: #
    json.dump(metadata, f)
