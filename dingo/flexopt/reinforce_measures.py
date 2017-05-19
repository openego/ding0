# reinforcement measures according to Ackermann
import os
import dingo
import pandas as pd
from dingo.tools import config as cfg_dingo
import logging

package_path = dingo.__path__[0]
logger = logging.getLogger('dingo')


def reinforce_branches_current(grid, crit_branches):
    #TODO: finish docstring
    """ Reinforce MV or LV grid by installing a new branch/line type
    
    Parameters
    ----------
        grid : GridDingo 
            Grid identifier.
        crit_branches : dict
            Dict of critical branches with max. relative overloading.

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
        
    Notes
    -----
        The branch type to be installed is determined per branch using the rel. overloading. According to [2]_ 
        only cables are installed.
        
    See Also
    --------
    dingo.flexopt.check_tech_constraints.check_load :
    dingo.flexopt.reinforce_measures.reinforce_branches_voltage :
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
    #TODO: finish docstring
    """ Reinforce MV or LV grid by installing a new branch/line type

    Parameters
    ----------
        grid : GridDingo 
            Grid identifier.
        crit_branches : :obj:`list` of :obj:`int`
            List of critical branches. #TODO: check if a list or a dictionary

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
        
    Notes
    -----
        The branch type to be installed is determined per branch - the next larger cable available is used.
        According to Ackermann only cables are installed.
        
    See Also
    --------
    dingo.flexopt.check_tech_constraints.check_load :
    dingo.flexopt.reinforce_measures.reinforce_branches_voltage :
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

    Parameters
    ----------
        grid : GridDingo 
            Grid identifier.

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    pass


def new_substation(grid):
    """ Reinforce MV grid by installing a new primary substation opposite to the existing one

    Parameters
    ----------
        grid : MVGridDingo 
            MV Grid identifier.

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly

    """