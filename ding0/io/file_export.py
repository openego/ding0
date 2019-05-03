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

from ding0.tools.results import load_nd_from_pickle
from ding0.io.export import export_network
from ding0.io.exporter_log import pickle_export_logger


def create_destination_dir():
    pass


def export_data_tocsv(path, network, csv_sep=','):
    # put a text file with the metadata
    metadata = json.loads(network.metadata_json)
    # can´t test this -> no permission for my user
    with open(os.path.join(path, 'metadata.json'), 'w') as metafile:
        json.dump(metadata, metafile)

    # Exports data to csv
    def export_network_tocsv(path, table, tablename):
        return table.to_csv(os.path.join(path, tablename + '.csv'), sep=csv_sep)

    export_network_tocsv(path, network.lv_grid, 'lv_grid')
    export_network_tocsv(path, network.lv_gen, 'lv_generator')
    export_network_tocsv(path, network.lv_cd, 'lv_branchtee')
    export_network_tocsv(path, network.lv_stations, 'lv_station')
    export_network_tocsv(path, network.mvlv_trafos, 'mvlv_transformer')
    export_network_tocsv(path, network.lv_loads, 'lv_load')
    export_network_tocsv(path, network.mv_grid, 'mv_grid')
    export_network_tocsv(path, network.mv_gen, 'mv_generator')
    export_network_tocsv(path, network.mv_cd, 'mv_branchtee')
    export_network_tocsv(path, network.mv_stations, 'mv_station')
    export_network_tocsv(path, network.hvmv_trafos, 'hvmv_transformer')
    export_network_tocsv(path, network.mv_cb, 'mv_circuitbreaker')
    export_network_tocsv(path, network.mv_loads, 'mv_load')
    export_network_tocsv(path, network.lines, 'line')
    export_network_tocsv(path, network.mvlv_mapping, 'mvlv_mapping')
    # export_network_tocsv(path, areacenter, 'areacenter')


if __name__ == '__main__':
    """
    Advise:
    First off check for existing .csv files in your destination folder. 
    Existing files will be extended.
    Multiple grids will be stored all in one file. 
    """

    # Path to user dir, Log file for missing Grid_Districts, Will be crated if not existing
    LOG_FILE_PATH = os.path.join(os.path.expanduser("~"), '.ding0_log', 'pickle_log')
    pickle_export_logger(LOG_FILE_PATH)

    # static path
    # Insert your own path
    pkl_filepath = "/home/local/RL-INSTITUT/jonas.huber/rli/Daten_flexibel_01/Ding0/20180823154014"

    # static path, .csv will be stored here
    # change this to your own destination folder
    destination_path = pkl_filepath

    # choose MV Grid Districts to import use list of integers
    # Multiple grids f. e.: grids = list(range(1, 3609)) - 1 to 3608(all of the existing)
    # Single grids f. e.: grids = [2]
    grids = list(range(1, 3))

    # Loop over all selected Grids, exports every singele one to file like .csv
    for grid_no in grids:

        try:
            nw = load_nd_from_pickle(os.path.join(pkl_filepath, 'ding0_grids__{}.pkl'.format(grid_no)))
        except:
            print('Something went wrong, created log entry in: {}'.format(LOG_FILE_PATH))
            with open(LOG_FILE_PATH, 'a') as log:
                log.write('ding0_grids__{}.pkl not present to the current directory\n'.format(grid_no))
                pass

            continue

        # Extract data from network and create DataFrames
        # pandas DataFrames will be exported as .csv file
        network_tupels = export_network(nw, run_id=nw.metadata['run_id'])
        export_data_tocsv(destination_path, network_tupels)

