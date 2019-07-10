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


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.

    Parameters
    ----------
    x: dict
    y: dict

    Note
    -----
    This function was originally proposed by
    http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression

    Credits to Thomas Vander Stichele. Thanks for sharing ideas!

    Returns
    -------
    :obj:`dict`
        Merged dictionary keyed by top-level keys of both dicts
    '''

    z = x.copy()
    z.update(y)
    return z