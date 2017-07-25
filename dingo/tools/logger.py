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
cfg_ding0.load_config('config_files.cfg')

import os
import logging


def create_dir(dirpath):
    """
    Create directory and report about it

    Parameters
    ----------
    dirpath : str
        Directory including path
    """

    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)

        print("We create a directory for you and your Ding0 data: {}".format(
            dirpath))


def create_home_dir(ding0_path=None):
    """
    Check in ~/<DING0_DIR> exists, otherwise create it

    Parameters
    ----------
    ding0_path : str
        Path to store Ding0 related data (logging, etc)
    """

    if ding0_path is None:
        ding0_path = get_default_home_dir()

    create_dir(ding0_path)


def get_default_home_dir():
    """
    Return default home directory of Ding0

    Returns
    -------
    homedir : str
        Default home directory including its path
    """
    ding0_dir = str(cfg_ding0.get('config',
                                  'config_dir'))
    return os.path.join(os.path.expanduser('~'), ding0_dir)


def setup_logger(log_dir=None, loglevel=logging.DEBUG):
    """
    Instantiate logger

    Parameters
    ----------
    log_dir : str
        Directory to save log, default: ~/.ding0/logging/
    """

    create_home_dir()
    create_dir(os.path.join(get_default_home_dir(), 'log'))

    if log_dir is None:
        log_dir = os.path.join(get_default_home_dir(), 'log')

    logger = logging.getLogger('ding0') # use filename as name in log
    logger.setLevel(loglevel)

    # create a file handler
    handler = logging.FileHandler(os.path.join(log_dir, 'ding0.log'))
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s-%(funcName)s-%(message)s (%(levelname)s)')
    handler.setFormatter(formatter)

    # create a stream handler (print to prompt)
    stream = logging.StreamHandler()
    stream.setLevel(logging.INFO)
    stream_formatter = logging.Formatter(
        '%(message)s (%(levelname)s)')
    stream.setFormatter(stream_formatter)

    # add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(stream)

    logger.info('########## New run of Ding0 issued #############')

    return logger
