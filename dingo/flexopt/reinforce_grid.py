from .check_tech_constraints import check_load, check_voltage


def reinforce_grid(grid, mode):
    """ Evaluates grid reinforcement needs and performs measures
    Args:
        grid: GridDingo object
        mode: kind of grid ('MV' or 'LV')

    Returns:

    Notes:

    References:
    .. [1] dena VNS

    """

    # which kind of grid is to be evaluated?
    if mode == 'MV':
        crit_branches, crit_stations = check_load(grid, mode)
        print(crit_branches)

        # mark critical branches
        for branch in crit_branches:
            branch['branch'].critical = True

        crit_nodes = check_voltage(grid, mode)
        grid.graph_draw()
        print(crit_nodes)
    elif mode == 'LV':
        check_load(grid, mode)



    #nodes = grid.
    #check_voltage(grid, mode, nodes)


