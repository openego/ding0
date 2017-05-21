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


from .check_tech_constraints import check_load, check_voltage, \
    get_critical_line_loading, get_critical_voltage_at_nodes
from .reinforce_measures import reinforce_branches_current, \
    reinforce_branches_voltage, reinforce_lv_branches_overloading, \
    extend_substation
import logging


logger = logging.getLogger('dingo')


def reinforce_grid(grid, mode):
    """ Evaluates grid reinforcement needs and performs measures. This function
        is the parent function for all grid reinforcements.

    Parameters
    ----------
    grid: GridDingo object
    mode: String
        kind of grid ('MV' or 'LV')

    Notes
    -----
    Currently only MV branch reinforcement is implemented. HV-MV stations are not
    reinforced since not required for status-quo scenario.

    References
    ----------
    .. [1] dena VNS
    .. [2] Ackermann et al. (RP VNS)

    """

    # kind of grid to be evaluated (MV or LV)
    if mode == 'MV':
        crit_branches, crit_stations = check_load(grid, mode)

        # STEP 1: reinforce branches

        # do reinforcement
        reinforce_branches_current(grid, crit_branches)

        # if branches or stations have been reinforced: run PF again to check for voltage issues
        if crit_branches or crit_stations:
            grid.network.run_powerflow(conn=None, method='onthefly')

        crit_nodes = check_voltage(grid, mode)
        crit_nodes_count_prev_step = len(crit_nodes)

        # as long as there are voltage issues, do reinforcement
        while crit_nodes:
            # determine all branches on the way from HV-MV substation to crit. nodes
            crit_branches_v = grid.find_and_union_paths(grid.station(), crit_nodes)

            # do reinforcement
            reinforce_branches_voltage(grid, crit_branches_v)

            # run PF
            grid.network.run_powerflow(conn=None, method='onthefly')

            crit_nodes = check_voltage(grid, mode)

            # if there are critical nodes left but no larger cable available, stop reinforcement
            if len(crit_nodes) == crit_nodes_count_prev_step:
                logger.warning('==> There are {0} branches that cannot be '
                               'reinforced (no appropriate cable '
                               'available).'.format(
                    len(grid.find_and_union_paths(grid.station(),
                                                        crit_nodes))))
                break

            crit_nodes_count_prev_step = len(crit_nodes)

        if not crit_nodes:
            logger.info('==> All voltage issues in {mode} grid could be '
                        'solved using reinforcement.'.format(mode=mode))

        # STEP 2: reinforce HV-MV station
        # NOTE: HV-MV station reinforcement is not required for status-quo
        # scenario since HV-MV trafos already sufficient for load+generation
        # case as done in MVStationDingo.choose_transformers()

    elif mode == 'LV':
        # get overloaded branches
        # overloading issues
        critical_branches, critical_stations = get_critical_line_loading(grid)


        # reinforce overloaded lines by increasing size
        unresolved = reinforce_lv_branches_overloading(grid, critical_branches)
        logger.info(
            "Out of {crit_branches} with overloading {unresolved} remain "
            "with unresolved issues due to line overloading. "
            "LV grid: {grid}".format(
                crit_branches=len(critical_branches),
                unresolved=len(unresolved),
                grid=grid))

        # reinforce substations
        extend_substation(grid, critical_stations, mode)

        # get node with over-voltage
        crit_nodes = get_critical_voltage_at_nodes(grid) #over-voltage issues

        crit_nodes_count_prev_step = len(crit_nodes)

        logger.info('{cnt_crit_branches} in {grid} have voltage issues'.format(
            cnt_crit_branches=crit_nodes_count_prev_step,
            grid=grid))

        # as long as there are voltage issues, do reinforcement
        while crit_nodes:
            # determine all branches on the way from HV-MV substation to crit. nodes
            crit_branches_v = grid.find_and_union_paths(
                grid.station(),
                [_['node'] for _ in crit_nodes])

            # do reinforcement
            reinforce_branches_voltage(grid, crit_branches_v, mode)

            # get node with over-voltage
            crit_nodes = get_critical_voltage_at_nodes(grid)

            # if there are critical nodes left but no larger cable available, stop reinforcement
            if len(crit_nodes) == crit_nodes_count_prev_step:
                logger.warning('==> There are {0} branches that cannot be '
                               'reinforced (no appropriate cable '
                               'available).'.format(
                    len(crit_branches_v)))
                break

            crit_nodes_count_prev_step = len(crit_nodes)

        if not crit_nodes:
            logger.info('==> All voltage issues in {mode} grid could be '
                        'solved using reinforcement.'.format(mode=mode))

