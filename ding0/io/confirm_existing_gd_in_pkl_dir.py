import os
import logging
from ding0.tools.results import load_nd_from_pickle

# LOG_FILE_PATH = 'pickle_log'
LOG_FILE_PATH = os.path.join(os.path.expanduser("~"), '.ding0_log', 'pickle_log')

# does the file exist?
if not os.path.isfile(LOG_FILE_PATH):
    print('ding0 log-file {file} not found. '
          'This might be the first run of the tool. '
          .format(file=LOG_FILE_PATH))
    base_path = os.path.split(LOG_FILE_PATH)[0]
    if not os.path.isdir(base_path):
        os.mkdir(base_path)
        print('The directory {path} was created.'.format(path=base_path))

    with open(LOG_FILE_PATH, 'a') as log:
        log.write("List of missing grid districts:")
        pass


# logging.basicConfig(filename=LOG_FILE_PATH, level=logging.DEBUG)

# pickle file locations path to RLI_Daten_Flex01 mount
pkl_filepath = "/home/local/RL-INSTITUT/jonas.huber/rli/Daten_flexibel_01/Ding0/20180823154014"

# choose MV Grid Districts to import
grids = list(range(61, 70))

for grid_no in grids:
    try:
        nw = load_nd_from_pickle(os.path.join(pkl_filepath, 'ding0_grids__{}.pkl'.format(grid_no)))
    except:
        # logging.debug('ding0_grids__{}.pkl not present to the current directory'.format(grid_no))
        with open(LOG_FILE_PATH, 'a') as log:
            log.write('ding0_grids__{}.pkl not present to the current directory\n'.format(grid_no))
            pass