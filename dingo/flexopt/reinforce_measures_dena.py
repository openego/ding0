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


# reinforcement measures according to dena


def parallel_branch(grid, node_target):
    """ Reinforce MV or LV grid by installing a new parallel branch

    Args:
        grid: GridDingo object
        node_target: node where the parallel cable (starting from HV/MV substation) is connected to
                     (used when grid is a MV grid)

    Returns:

    """
    pass


def split_ring(grid):
    """ Reinforce MV grid by splitting a critical ring into two new rings

    Args:
        grid: MVGridDingo object

    Returns:

    """
    pass


def extend_substation(grid):
    """ Reinforce MV or LV substation by exchanging the existing trafo and installing a parallel one if necessary with

    Args:
        grid: GridDingo object

    Returns:

    """
    pass


def new_substation(grid):
    """ Reinforce MV grid by installing a new primary substation opposite to the existing one

    Args:
        grid: MVGridDingo object

    Returns:

    """