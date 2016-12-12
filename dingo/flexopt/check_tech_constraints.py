# check technical constraints of distribution grids (shared lib)

from dingo.tools import config as cfg_dingo


def check_load(grid, mode):
    """ Checks for over-loading of branches and transformers for MV or LV grid

    Args:
        grid: GridDingo object
        mode: kind of grid ('MV' or 'LV')

    Returns:
        List of critical branches and transformers,
        Format: [branch_1, ..., branch_n], [trafo_1, ..., trafo_m]

    Notes:
        Lines'/cables' max. capacity (load case and feed-in case) are taken from [1]_.

    References:
    .. [1] dena VNS

    """

    crit_branches = []
    crit_trafos = []

    if mode == 'MV':
        load_factor_mv_trans_lc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_trans_lc_normal'))
        load_factor_mv_line_lc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_line_lc_normal'))
        load_factor_mv_cable_lc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_cable_lc_normal'))
        load_factor_mv_trans_fc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_trans_fc_normal'))
        load_factor_mv_line_fc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_line_fc_normal'))
        load_factor_mv_cable_fc_normal = float(cfg_dingo.get('assumptions',
                                                             'load_factor_mv_cable_fc_normal'))

        mw2kw = 1e3

        # check branches' loads
        for branch in grid.graph_edges():
            s_max_th = 3**0.5 * branch['branch'].type['U_n'] * branch['branch'].type['I_max_th']
            # TODO: Add type attribute to branch for checking type !!!!
            if grid.v_level == 20:
                s_max_th = s_max_th * load_factor_mv_line_lc_normal
            elif grid.v_level == 10:
                s_max_th = s_max_th * load_factor_mv_cable_lc_normal

            if any(s*mw2kw >= s_max_th for s in branch['branch'].s_res):
                crit_branches.append(branch)

        # check trafos' loads
        for trafo in grid.graph_edges():
            s_max_th = 3**0.5 * branch['branch'].type['U_n'] * branch['branch'].type['I_max_th']
            if any(s*mw2kw >= s_max_th for s in branch['branch'].s_res):
                crit_trafos.append(trafo)

    elif mode == 'LV':
        pass


def check_voltage(grid, mode, nodes):
    """ Checks for voltage stability issues at all nodes for MV or LV grid

    Args:
        grid: GridDingo object
        mode: kind of grid ('MV' or 'LV')
        nodes: List of members

    Returns:
        List of critical nodes, sorted descending by voltage difference

    Notes:
        The examination is done in two steps, according to [1]_:
        1. It is checked

    References:
    .. [1] dena VNS
    """

    if mode == 'MV':
        pass
    elif mode == 'LV':
        pass





