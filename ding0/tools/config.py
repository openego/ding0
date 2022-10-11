"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io

Based on code by oemof development team

This module provides a highlevel layer for reading and writing config files.
The config file has to be of the following structure to be imported correctly.

::

    [netCDF] \n
         RootFolder = c://netCDF \n
         FilePrefix = cd2_ \n
    \n
    [mySQL] \n
        host = localhost \n
        user = guest \n
        password = root \n
        database = znes \n
    \n
    [SectionName] \n
        OptionName = value \n
        Option2 = value2 \n

Based on code by oemof development team
"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


import configparser as cp
import logging
import os.path as path

import ding0

logger = logging.getLogger(__name__)

cfg = cp.RawConfigParser()
_loaded = False

def load_config(filename):
    """ Read config file specified by `filename`
    
    Parameters
    ----------
    filename : :obj:`str`
        Description of filename
    """
    package_path = ding0.__path__[0]
    FILE = path.join(package_path, 'config', filename)

    try:
        cfg.read(FILE)
        global _loaded
        _loaded = True
    except:
        logger.exception("configfile not found.")

def get(section, key):
    """Returns the value of a given key of a given section of the main
    config file.
    
    Parameters
    ----------
    section : :obj:`str`
        the section.
    key : :obj:`str`
        the key
    
    Returns
    -------
    :any:`float`
        the value which will be casted to float, int or boolean.
        if no cast is successful, the raw string will be returned.
        
    See Also
    --------
    set :
    """
    if not _loaded:
        pass
    try:
        return cfg.getfloat(section, key)
    except Exception:
        try:
            return cfg.getint(section, key)
        except:
            try:
                return cfg.getboolean(section, key)
            except:
                return cfg.get(section, key)


def set(section, key, value):
    """Sets a value to a [section] key - pair.
    
    if the section doesn't exist yet, it will be created.
    
    Parameters
    ----------
    section: :obj:`str`
        the section.
    key: :obj:`str`
        the key.
    value: float, int, str
        the value.
        
    See Also
    --------
    get :
    """

    if not _loaded:
        init()

    if not cfg.has_section(section):
        cfg.add_section(section)

    cfg.set(section, key, value)

    with open(FILE, 'w') as configfile:
        cfg.write(configfile)
