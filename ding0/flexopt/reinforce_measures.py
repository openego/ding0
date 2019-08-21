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


# reinforcement measures according to Ackermann
import os
import ding0
import pandas as pd
from ding0.tools import config as cfg_ding0
from ding0.grid.lv_grid.build_grid import select_transformers
from ding0.core.network import TransformerDing0
from ding0.flexopt.check_tech_constraints import get_voltage_at_bus_bar
import networkx as nx
import logging

package_path = ding0.__path__[0]
logger = logging.getLogger('ding0')


def reinforce_branches_current(grid, crit_branches):
    #TODO: finish docstring
    """ Reinforce MV or LV grid by installing a new branch/line type
    
    Parameters
    ----------
    grid : :class:`~.ding0.core.GridDing0`
        Grid identifier.
    crit_branches : dict
        Dict of critical branches with max. relative overloading.
        
    Note
    -----
    The branch type to be installed is determined per branch using the rel. overloading. According to [#]_ 
    only cables are installed.
    
    References
    ----------
    .. [#] Ackermann et al. (RP VNS)    
    
    See Also
    --------
    ding0.flexopt.check_tech_constraints.check_load :
    ding0.flexopt.reinforce_measures.reinforce_branches_voltage :
    """
    # load cable data, file_names and parameter
    branch_parameters = grid.network.static_data['MV_cables']
    branch_parameters = branch_parameters[branch_parameters['U_n'] == grid.v_level].sort_values('I_max_th')

    branch_ctr = 0

    for branch, rel_overload in crit_branches.items():
        try:
            type = branch_parameters.loc[
                branch_parameters[
                    branch_parameters['I_max_th'] >= branch['branch']
                    .type['I_max_th'] * rel_overload
                ].loc[
                    :, 'I_max_th'
                ].idxmin(), :
            ]
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
    #TODO: finish docstring
    """ Reinforce MV or LV grid by installing a new branch/line type

    Parameters
    ----------
    grid : :class:`~.ding0.core.GridDing0`
        Grid identifier.
    crit_branches : :obj:`list` of :obj:`int`
        List of critical branches. #TODO: check if a list or a dictionary
    grid_level : :obj:`str`
        Specifying either 'MV' for medium-voltage grid or 'LV' for
        low-voltage grid level.
        
    Note
    -----
    The branch type to be installed is determined per branch - the next larger cable available is used.
    According to Ackermann only cables are installed.
        
    See Also
    --------
    ding0.flexopt.check_tech_constraints.check_load :
    ding0.flexopt.reinforce_measures.reinforce_branches_voltage :
    """

    # load cable data, file_names and parameter
    branch_parameters = grid.network.static_data['{gridlevel}_cables'.format(
        gridlevel=grid_level)]
    branch_parameters = branch_parameters[branch_parameters['U_n'] == grid.v_level].sort_values('I_max_th')

    branch_ctr = 0

    for branch in crit_branches:
        try:
            type = branch_parameters.loc[
                branch_parameters.loc[
                       branch_parameters['I_max_th'] > branch.type['I_max_th']
                ].loc[
                    :, 'I_max_th'
                ].idxmin(), :
            ]
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
    """
    Reinforce MV or LV substation by exchanging the existing trafo and
    installing a parallel one if necessary.

    First, all available transformers in a `critical_stations` are extended to
    maximum power. If this does not solve all present issues, additional
    transformers are build.

    Parameters
    ----------
    grid: :class:`~.ding0.core.GridDing0`
        Ding0 grid container
    critical_stations : :obj:`list`
        List of stations with overloading
    grid_level : :obj:`str`
        Either "LV" or "MV". Basis to select right equipment.
    
    Note
    -----
    Curently straight forward implemented for LV stations

    Returns
    -------
    type 
        #TODO: Description of return. Change type in the previous line accordingly
    """
    load_factor_lv_trans_lc_normal = cfg_ding0.get(
        'assumptions',
        'load_factor_lv_trans_lc_normal')
    load_factor_lv_trans_fc_normal = cfg_ding0.get(
        'assumptions',
        'load_factor_lv_trans_fc_normal')

    trafo_params = grid.network._static_data['{grid_level}_trafos'.format(
        grid_level=grid_level)]
    trafo_s_max_max = max(trafo_params['S_nom'])


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
            # only work with first of potentially multiple trafos
            trafo = extendable_trafos[0]
            trafo_s_max_a_before = trafo.s_max_a

            # extend power of first trafo to next higher size available
            extend_trafo_power(extendable_trafos, trafo_params)

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
                lv_transformer = TransformerDing0(
                    grid=grid,
                    id_db=id,
                    v_level=0.4, #ToDo: replace hard coded number by meta v_nom
                    s_max_longterm=trafo_type['S_nom'],
                    r=trafo_type['R'],
                    x=trafo_type['X'])

                # add each transformer to its station
                grid._station.add_transformer(lv_transformer)

    logger.info("{stations_cnt} have been reinforced due to overloading "
                "issues.".format(stations_cnt=len(critical_stations)))


def extend_substation_voltage(crit_stations, grid_level='LV'):
    """
    Extend substation if voltage issues at the substation occur

    Follows a two-step procedure:
    
    i) Existing transformers are extended by replacement with large nominal
       apparent power
      
    ii) New additional transformers added to substation (see 'Note')

    Parameters
    ----------
    crit_stations : :obj:`list`
        List of stations with overloading or voltage issues.
    grid_level : :obj:`str`
        Specifiy grid level: 'MV' or 'LV'

    Note
    -----
    At maximum 2 new of largest (currently 630 kVA) transformer are additionally
    built to resolve voltage issues at MV-LV substation bus bar.
    """
    grid = crit_stations[0]['node'].grid
    trafo_params = grid.network._static_data['{grid_level}_trafos'.format(
        grid_level=grid_level)]
    trafo_s_max_max = max(trafo_params['S_nom'])
    trafo_min_size = trafo_params.loc[trafo_params['S_nom'].idxmin(), :]

    v_diff_max_fc = cfg_ding0.get('assumptions', 'lv_max_v_level_fc_diff_normal')
    v_diff_max_lc = cfg_ding0.get('assumptions', 'lv_max_v_level_lc_diff_normal')

    tree = nx.dfs_tree(grid._graph, grid._station)

    for station in crit_stations:
        v_delta = max(station['v_diff'])

        # get list of nodes of main branch in right order
        extendable_trafos = [_ for _ in station['node']._transformers
                             if _.s_max_a < trafo_s_max_max]

        v_delta_initially_lc = v_delta[0]
        v_delta_initially_fc = v_delta[1]

        new_transformers_cnt = 0

        # extend existing trafo power while voltage issues exist and larger trafos
        # are available
        while (v_delta[0] > v_diff_max_lc) or (v_delta[1] > v_diff_max_fc):
            if extendable_trafos:
                # extend power of first trafo to next higher size available
                extend_trafo_power(extendable_trafos, trafo_params)
            elif new_transformers_cnt < 2:
                # build a new transformer
                lv_transformer = TransformerDing0(
                    grid=grid,
                    id_db=id,
                    v_level=0.4, #ToDo: replace hardcoded number by meta v_nom
                    s_max_longterm=trafo_min_size['S_nom'],
                    r=trafo_min_size['R'],
                    x=trafo_min_size['X'])

                # add each transformer to its station
                grid._station.add_transformer(lv_transformer)

                new_transformers_cnt += 1

            # update break criteria
            v_delta = get_voltage_at_bus_bar(grid, tree)
            extendable_trafos = [_ for _ in station['node']._transformers
                                 if _.s_max_a < trafo_s_max_max]

            if (v_delta[0] == v_delta_initially_lc) or (
                v_delta[1] == v_delta_initially_fc):
                logger.warning("Extension of {station} has no effect on "
                               "voltage delta at bus bar. Transformation power "
                               "extension is halted.".format(
                    station=station['node']))
                break

        if (v_delta[0] > v_diff_max_lc) or (v_delta[1] > v_diff_max_fc):
            raise Exception ('Voltage issue at substation {} could not be resolved. LV grid reinforcement'
                             'can therefore never result in correct grids. Check MV grid for line reinforcement.'.format(station))


def new_substation(grid):
    """ Reinforce MV grid by installing a new primary substation opposite to the existing one

    Parameters
    ----------
    grid : :class:`~.ding0.core.network.grids.MVGridDing0`
        MV Grid identifier.
    """


def reinforce_lv_branches_overloading(grid, crit_branches):
    """
    Choose appropriate cable type for branches with line overloading

    Parameters
    ----------
    grid : :class:`~.ding0.core.network.grids.LVGridDing0`
        Ding0 LV grid object
    crit_branches : :obj:`list`
        List of critical branches incl. its line loading

    Note
    -----
    If maximum size cable is not capable to resolve issue due to line
    overloading largest available cable type is assigned to branch.

    Returns
    -------
    :obj:`list`
        unsolved_branches : List of braches no suitable cable could be found
    """
    unsolved_branches = []

    cable_lf = cfg_ding0.get('assumptions',
                             'load_factor_lv_cable_lc_normal')

    cables = grid.network.static_data['LV_cables']

    # resolve overloading issues for each branch segment
    for branch in crit_branches:
        I_max_branch_load = branch['s_max'][0]/(3**0.5 * grid.v_level / 1e3)
        I_max_branch_gen = branch['s_max'][1]/(3**0.5 * grid.v_level / 1e3)
        I_max_branch = max([I_max_branch_load, I_max_branch_gen])

        suitable_cables = cables[(cables['I_max_th'] * cable_lf)
                          > I_max_branch]

        if not suitable_cables.empty:
            cable_type = suitable_cables.loc[suitable_cables['I_max_th'].idxmin(), :]
            branch['branch'].type = cable_type
            crit_branches.remove(branch)
        else:
            cable_type_max = cables.loc[cables['I_max_th'].idxmax(), :]
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


def extend_trafo_power(extendable_trafos, trafo_params):
    """
    Extend power of first trafo in list of extendable trafos

    Parameters
    ----------
    extendable_trafos : :obj:`list`
        Trafos with rated power below maximum size available trafo
    trafo_params : :pandas:`pandas.DataFrame<dataframe>`
        Transformer parameters
    """
    trafo = extendable_trafos[0]
    trafo_s_max_a_before = trafo.s_max_a
    trafo_nearest_larger = trafo_params.loc[
        trafo_params.loc[
            trafo_params['S_nom'] > trafo_s_max_a_before
        ].loc[
            :, 'S_nom'
        ].idxmin(), :
    ]
    trafo.s_max_a = trafo_nearest_larger['S_nom']
    trafo.r = trafo_nearest_larger['R']
    trafo.x = trafo_nearest_larger['X']
