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
from dingo.tools import config as cfg_dingo
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


def reinforce_lv_branches_overloading(grid, crit_branches):
    """
    Choose appropriate cable type for branches with line overloading

    Parameters
    ----------
    grid : dingo.core.network.grids.LVGridDingo
        Dingo LV grid object
    crit_branches : list
        List of critical branches incl. its line loading

    Notes
    -----
    If maximum size cable is not capable to resolve issue due to line
    overloading largest available cable type is assigned to branch.

    Returns
    -------

        unsolved_branches : :obj:`list`
            List of braches no suitable cable could be found
    """
    unsolved_branches = []

    cable_lf = cfg_dingo.get('assumptions',
                             'load_factor_lv_cable_lc_normal')

    cables = grid.network.static_data['LV_cables']

    # resolve overloading issues for each branch segment
    for branch in crit_branches:
        I_max_branch_load = branch['s_max'][0]
        I_max_branch_gen = branch['s_max'][1]
        I_max_branch = max([I_max_branch_load, I_max_branch_gen])

        suitable_cables = cables[(cables['I_max_th'] * cable_lf)
                          > I_max_branch]

        if not suitable_cables.empty:
            cable_type = suitable_cables.ix[suitable_cables['I_max_th'].idxmin()]
            branch['branch'].type = cable_type
            crit_branches.remove(branch)
        else:
            cable_type_max = cables.ix[cables['I_max_th'].idxmax()]
            unsolved_branches.append(branch)
            branch['branch'].type = cable_type_max
            logger.error("No suitable cable type could be found for {branch} "
                          "with I_th_max = {current}. "
                          "Cable of type {cable} is chosen during "
                          "reinforcement.".format(
                branch=branch['branch'],
                cable=cable_type_max.name,
                current=I_max_branch
            ))

    return unsolved_branches