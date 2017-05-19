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

        print("We create a directory for you and your Dingo data: {}".format(
            dirpath))


def create_home_dir(dingo_path=None):
    """
    Check in ~/.dingo exists, otherwise create it

    Parameters
    ----------
    dingo_path : str
        Path to store Dingo related data (logging, etc)
    """

    if dingo_path is None:
        dingo_path = get_default_home_dir()

    create_dir(dingo_path)


def get_default_home_dir():
    """
    Return default home directory of Dingo

    Returns
    -------
    homedir : str
        Default home directory including its path
    """
    return os.path.join(os.path.expanduser('~'), '.dingo')


def setup_logger(log_dir=None):
    """
    Instantiate logger

    Parameters
    ----------
    log_dir : str
        Directory to save log, default: ~/.dingo/logging/
    """

    create_home_dir()
    create_dir(os.path.join(get_default_home_dir(), 'log'))

    if log_dir is None:
        log_dir = os.path.join(get_default_home_dir(), 'log')

    logger = logging.getLogger('dingo') # use filename as name in log
    logger.setLevel(logging.DEBUG)

    # create a file handler
    handler = logging.FileHandler(os.path.join(log_dir, 'dingo.log'))
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

    logger.info('########## New run of Dingo issued #############')

    return logger