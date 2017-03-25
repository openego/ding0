from .check_tech_constraints import check_load, check_voltage
from .reinforce_measures import reinforce_branches_current, reinforce_branches_voltage, extend_substation, new_substation


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
        #print(crit_branches)

        # mark critical branches
        #for branch in crit_branches:
        #    branch['branch'].critical = True

        # do reinforcement
        reinforce_branches_current(grid, crit_branches)

        # run PF again to check for voltage issues
        grid.network.run_powerflow(conn=None, method='onthefly')

        crit_nodes = check_voltage(grid, mode)
        #print(crit_nodes)

        # determine all branches on the way from HV-MV substation to crit. nodes
        crit_branches_v = grid.find_and_union_paths(grid.station(), crit_nodes)

        # do reinforcement
        reinforce_branches_voltage(grid, crit_branches_v)
        # grid.graph_draw()

        grid.network.run_powerflow(conn=None, method='onthefly')
        crit_branches, crit_stations = check_load(grid, mode)
        crit_nodes = check_voltage(grid, mode)


    elif mode == 'LV':
        raise NotImplementedError
        #check_load(grid, mode)



    #nodes = grid.
    #check_voltage(grid, mode, nodes)


