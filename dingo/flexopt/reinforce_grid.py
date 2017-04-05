from .check_tech_constraints import check_load, check_voltage
from .reinforce_measures import reinforce_branches_current, reinforce_branches_voltage, extend_substation, new_substation
import logging


logger = logging.getLogger('dingo')


def reinforce_grid(grid, mode):
    """ Evaluates grid reinforcement needs and performs measures
    Args:
        grid: GridDingo object
        mode: kind of grid ('MV' or 'LV')

    Returns:

    Notes:

    References:
    .. [1] dena VNS
    .. [2] Ackermann et al. (RP VNS)

    """

    # kind of grid to be evaluated (MV or LV)
    if mode == 'MV':
        crit_branches, crit_stations = check_load(grid, mode)

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
            logger.info('==> All voltage issues could be solved using '
                        'reinforcement.')

    elif mode == 'LV':
        raise NotImplementedError
