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


# reinforcement measures according to Ackermann
import os
import dingo
import pandas as pd
import logging

package_path = dingo.__path__[0]
logger = logging.getLogger('dingo')


def reinforce_branches_current(grid, crit_branches):
    """ Reinforce MV or LV grid by installing a new branch/line type

    Args:
        grid: GridDingo object
        crit_branches; Dict of critical branches with max. relative overloading

    Notes:
        The branch type to be installed is determined per branch using the rel. overloading. According to Ackermann
        only cables are installed.
    """
    # load cable data, file_names and parameter
    branch_parameters = grid.network.static_data['MV_cables']
    branch_parameters = branch_parameters[branch_parameters['U_n'] == grid.v_level].sort_values('I_max_th')

    branch_ctr = 0

    for branch, rel_overload in crit_branches.items():
        try:
            type = branch_parameters.ix[branch_parameters[branch_parameters['I_max_th'] >=
                                        branch['branch'].type['I_max_th'] * rel_overload]['I_max_th'].idxmin()]
            branch['branch'].type = type
            branch_ctr += 1
        except:
            logger.warning('Branch {} could not be reinforced (current '
                           'issues) as there is no appropriate cable type '
                           'available. Original type is retained.'.format(
                branch))
            pass

    if branch_ctr:
        logger.info('==> {} branches were reinforced.'.format(str(branch_ctr)))


def reinforce_branches_voltage(grid, crit_branches):
    """ Reinforce MV or LV grid by installing a new branch/line type

    Parameters
    ----------
    grid: GridDingo object
    crit_branches: List of BranchDingo objects
        list of critical branches

    Notes
    -----
    The branch type to be installed is determined per branch - the next larger cable
    available is used. According to [1]_ only cables are installed.

    References
    ----------
    .. [1] Ackermann et al. (RP VNS)
    """

    # load cable data, file_names and parameter
    branch_parameters = grid.network.static_data['MV_cables']
    branch_parameters = branch_parameters[branch_parameters['U_n'] == grid.v_level].sort_values('I_max_th')

    branch_ctr = 0

    for branch in crit_branches:
        try:
            type = branch_parameters.ix[branch_parameters.loc[branch_parameters['I_max_th'] >
                                        branch.type['I_max_th']]['I_max_th'].idxmin()]
            branch.type = type
            branch_ctr += 1
        except:
            logger.warning('Branch {} could not be reinforced (voltage '
                           'issues) as there is no appropriate cable type '
                           'available. Original type is retained.'.format(
                branch))
            pass


    if branch_ctr:
        logger.info('==> {} branches were reinforced.'.format(str(branch_ctr)))


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