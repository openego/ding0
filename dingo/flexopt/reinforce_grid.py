from .check_tech_constraints import check_load, check_voltage
from .reinforce_measures import reinforce_branches, extend_substation, new_substation


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
        reinforce_branches(grid, crit_branches, mode='determined')

        # run PF again to check for voltage issues
        grid.nd.run_powerflow(conn=None, method='onthefly')

        crit_nodes = check_voltage(grid, mode)
        #print(crit_nodes)

        # determine all branches on the way from HV-MV substation to crit. nodes
        crit_branches_v = grid.find_and_union_paths(grid.station(), crit_nodes)

        # do reinforcement
        reinforce_branches(grid, crit_branches_v, mode='next')

        # grid.graph_draw()

    elif mode == 'LV':
        check_load(grid, mode)



    #nodes = grid.
    #check_voltage(grid, mode, nodes)


