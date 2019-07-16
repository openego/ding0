"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__ = "Reiner Lemoine Institut gGmbH"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__ = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__ = "jh-RLI"

import os

# LOG_FILE_PATH = 'pickle_log'
# LOG_FILE_PATH = os.path.join(os.path.expanduser("~"), '.ding0_log', 'pickle_log')

def pickle_export_logger(log_file_path):
    """
    Creates a list for pickle files that are missing for some reason.
    Most likely the file does not exists @ the pickle file path dir.

    Log missing ding0 GridDistricts:
    The export_log provides functionality that set ups a logging file. One need to provide a path to the destination
    the logfile shall be stored in.
    The file can be opened at any code line and input can be provided. This logger is mainly used within a exception
    to log the missing GridDistricts that could not be created by ding0.

    :param log_file_path:
    :return:
    """
    # does the file exist?
    if not os.path.isfile(log_file_path):
        print('ding0 log-file {file} not found. '
              'This might be the first run of the tool. '
              .format(file=log_file_path))
        base_path = os.path.split(log_file_path)[0]
        if not os.path.isdir(base_path):
            os.mkdir(base_path)
            print('The directory {path} was created.'.format(path=base_path))

        with open(log_file_path, 'a') as log:
            log.write("List of missing grid districts:")
            pass
