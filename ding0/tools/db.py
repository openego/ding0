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


from oemof.db import engine
from ding0.tools import config as cfg_ding0
from ding0.tools.logger import get_default_home_dir
import os

import logging
logger = logging.getLogger('ding0')


def connection(section='oedb'):
    """DING0-specific database connection method which uses oemof.db
    
    This function checks if DING0's config file for access to databases
    exists and provides functionality to create this file.
    
    This function purely calls the `connect()` method of the engine object
    returned by :py:func:`engine`.

    For description of parameters see :py:func:`engine`.
    """

    # get name of DB config file
    db_config_file = str(cfg_ding0.get('config',
                                       'db_config_file'))
    db_config_file_path = os.path.join(get_default_home_dir(),
                                       db_config_file)

    # does the file exist?
    if os.path.isfile(db_config_file_path):
        config_file = db_config_file_path

    # if not, ask to create with user input
    else:
        logger.warning('DB config file {} not found. '
                       'This might be the first run of DING0. '
                       'Do you want me to create this file? (y/n): '
                       .format(db_config_file_path))
        choice = ''
        while choice not in ['y', 'n']:
            choice = input('(y/n): ')

        if choice == 'y':
            username = input('Enter value for `username`: ')
            database = input('Enter value for `database`: ')
            host = input('Enter value for `host`: ')
            port = input('Enter value for `port` (default: 5432): ')

            file = open(db_config_file_path, 'w')
            template = '[{0}]\n' \
                       'username = {1}\n' \
                       'database = {2}\n' \
                       'host     = {3}\n' \
                       'port     = {4}\n'.format(section,
                                                 username,
                                                 database,
                                                 host,
                                                 '5432' if not port else port)
            file.write(template)
            file.close()

            logger.info('Template {0} with section `{1}` created.'
                        .format(db_config_file_path,
                                section))

            config_file = db_config_file_path

        # fallback: use oemof's config.ini
        else:
            logger.info('No DB config file created, I\'ll try to use oemof\'s config.ini')
            config_file=None

    return engine(section=section, config_file=config_file).connect()