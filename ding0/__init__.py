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


import numpy
from psycopg2.extensions import register_adapter, AsIs


# TODO: (Maybe) move to more general place (ego.io repo)
# TODO: check docstring
def adapt_numpy_int64(numpy_int64):
    """ Adapting numpy.int64 type to SQL-conform int type using psycopg extension, see [#]_ for more info.

    Parameters
    ----------
    numpy_int64 : int 
        numpty 64bits integer.
          
    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    
    References
    ----------
    
    .. [#] http://initd.org/psycopg/docs/advanced.html#adapting-new-python-types-to-sql-syntax
    """
    return AsIs(numpy_int64)


register_adapter(numpy.int64, adapt_numpy_int64)