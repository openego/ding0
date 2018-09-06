"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "nesnoj, gplssm"

import os

import json


def export_data_tocsv(path, run_id, metadata_json,
                      lv_grid, lv_gen, lv_cd, lv_stations, mvlv_trafos,
                      lv_loads,
                      mv_grid, mv_gen, mv_cb, mv_cd, mv_stations, hvmv_trafos,
                      mv_loads,
                      lines, mvlv_mapping, csv_sep=','):
    # make directory with run_id if it doesn't exist
    os.makedirs(path, exist_ok=True)

    # put a text file with the metadata
    metadata = json.loads(metadata_json)
    with open(os.path.join(path, 'metadata.json'), 'w') as metafile:
        json.dump(metadata, metafile)

    # Exports data to csv
    def export_network_tocsv(path, table, tablename):
        return table.to_csv(os.path.join(path, tablename + '.csv'), sep=csv_sep)

    export_network_tocsv(path, lv_grid, 'lv_grid')
    export_network_tocsv(path, lv_gen, 'lv_generator')
    export_network_tocsv(path, lv_cd, 'lv_branchtee')
    export_network_tocsv(path, lv_stations, 'lv_station')
    export_network_tocsv(path, mvlv_trafos, 'mvlv_transformer')
    export_network_tocsv(path, lv_loads, 'lv_load')
    export_network_tocsv(path, mv_grid, 'mv_grid')
    export_network_tocsv(path, mv_gen, 'mv_generator')
    export_network_tocsv(path, mv_cd, 'mv_branchtee')
    export_network_tocsv(path, mv_stations, 'mv_station')
    export_network_tocsv(path, hvmv_trafos, 'hvmv_transformer')
    export_network_tocsv(path, mv_cb, 'mv_circuitbreaker')
    export_network_tocsv(path, mv_loads, 'mv_load')
    export_network_tocsv(path, lines, 'line')
    export_network_tocsv(path, mvlv_mapping, 'mvlv_mapping')
    # export_network_tocsv(path, areacenter, 'areacenter')