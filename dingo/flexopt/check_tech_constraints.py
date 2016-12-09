# check technical constraints of distribution grids (shared lib)


def check_voltage(grid):
    """ Checks for voltage stability issues at all nodes for MV or LV grid

    Args:
        grid: GridDingo object

    Returns:
        List of critical nodes, sorted descending by voltage difference
    """
    pass


def check_load(grid):
    """ Checks for over-loading of branches for MV or LV grid

    Args:
        grid: GridDingo object

    Returns:
        List of critical branches
    """
    pass


