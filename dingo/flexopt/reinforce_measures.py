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
from dingo.grid.lv_grid.build_grid import select_transformers
from dingo.core.network import TransformerDingo
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


def reinforce_branches_voltage(grid, crit_branches, grid_level='MV'):
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
    branch_parameters = grid.network.static_data['{gridlevel}_cables'.format(
        gridlevel=grid_level)]
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


def extend_substation(grid, critical_stations, grid_level):
    """ Reinforce MV or LV substation by exchanging the existing trafo and
    installing a parallel one if necessary.

    First, all available transformers in a `critical_stations` are extended to
    maximum power. If this does not solve all present issues, additional
    transformers are build.

    Parameters
    ----------
        grid: GridDingo
            Dingo grid container
        critical_stations : list
            List of stations with overloading or voltage issues
        grid_level : str
            Either "LV" or "MV". Basis to select right equipment.

    Notes
    -----
    Curently straight forward implemented for LV stations

    """
    load_factor_lv_trans_lc_normal = cfg_dingo.get(
        'assumptions',
        'load_factor_lv_trans_lc_normal')
    load_factor_lv_trans_fc_normal = cfg_dingo.get(
        'assumptions',
        'load_factor_lv_trans_fc_normal')

    trafo_params = grid.network._static_data['{grid_level}_trafos'.format(
        grid_level=grid_level)]
    trafo_s_max_max = max(trafo_params['S_max'])


    for station in critical_stations:
        # determine if load or generation case and apply load factor
        if station['s_max'][0] > station['s_max'][1]:
            case = 'load'
            lf_lv_trans_normal = load_factor_lv_trans_lc_normal
        else:
            case = 'gen'
            lf_lv_trans_normal = load_factor_lv_trans_fc_normal


        # cumulative maximum power of transformers installed
        s_max_trafos = sum([_.s_max_a
                            for _ in station['station']._transformers])

        # determine missing trafo power to solve overloading issue
        s_trafo_missing = max(station['s_max']) - (
            s_max_trafos * lf_lv_trans_normal)

        # list of trafos with rated apparent power below `trafo_s_max_max`
        extendable_trafos = [_ for _ in station['station']._transformers
                             if _.s_max_a < trafo_s_max_max]

        # try to extend power of existing trafos
        while (s_trafo_missing > 0) and extendable_trafos:

            # extend power of first trafo to next higher size available
            trafo = extendable_trafos[0]
            trafo_s_max_a_before = trafo.s_max_a
            trafo_nearest_larger = trafo_params.ix[
                trafo_params.loc[trafo_params['S_max'] > trafo_s_max_a_before][
                    'S_max'].idxmin()]
            trafo.s_max_a = trafo_nearest_larger['S_max']
            trafo.r = trafo_nearest_larger['R']
            trafo.x = trafo_nearest_larger['X']

            # diminish missing trafo power by extended trafo power and update
            # extendable trafos list
            s_trafo_missing -= ((trafo.s_max_a * lf_lv_trans_normal) -
                                trafo_s_max_a_before)
            extendable_trafos = [_ for _ in station['station']._transformers
                                 if _.s_max_a < trafo_s_max_max]

        # build new trafos inside station until
        if s_trafo_missing > 0:
            trafo_type, trafo_cnt = select_transformers(grid, s_max={
                's_max': s_trafo_missing,
                'case': case
            })
            
            # create transformers and add them to station of LVGD
            for t in range(0, trafo_cnt):
                lv_transformer = TransformerDingo(
                    grid=grid,
                    id_db=id,
                    v_level=0.4,
                    s_max_longterm=trafo_type['S_max'],
                    r=trafo_type['R'],
                    x=trafo_type['X'])

                # add each transformer to its station
                grid._station.add_transformer(lv_transformer)

    logger.info("{stations_cnt} have been reinforced due to overloading "
                "issues.".format(stations_cnt=len(critical_stations)))


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