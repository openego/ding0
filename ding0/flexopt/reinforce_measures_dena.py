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


# reinforcement measures according to dena


def parallel_branch(grid, node_target):
    """ Reinforce MV or LV grid by installing a new parallel branch according to dena

    Parameters
    ----------
        grid : :class:`~.ding0.core.GridDing0`
            Grid identifier.
        node_target: :obj:`int`
            node where the parallel cable (starting from HV/MV substation) is connected to (used when grid is a MV grid)

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    pass


def split_ring(grid):
    """ Reinforce MV grid by splitting a critical ring into two new rings according to dena
    
    Parameters
    ----------
        grid : :class:`~.ding0.core.network.grids.MVGridDing0`
            Grid identifier.
          
    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    pass


def extend_substation(grid):
    """ Reinforce MV or LV substation by exchanging the existing trafo and installing a parallel one if necessary with according to dena

    Parameters
    ----------
        grid : :class:`~.ding0.core.GridDing0`
            Grid identifier.

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    pass


def new_substation(grid):
    """ Reinforce MV grid by installing a new primary substation opposite to the existing one according to dena

    Parameters
    ----------
        grid : :class:`~.ding0.core.network.grids.MVGridDing0`
            Grid identifier.
            
    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
