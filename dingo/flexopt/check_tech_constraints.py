# check technical constraints of distribution grids (shared lib)

from dingo.tools import config as cfg_dingo
import logging


logger = logging.getLogger('dingo')


def check_load(grid, mode):
    """ Checks for over-loading of branches and transformers for MV or LV grid

    Args:
        grid: GridDingo object
        mode: kind of grid ('MV' or 'LV')

    Returns:
        Dict of critical branches with max. relative overloading
        List of critical transformers,
        Format: {branch_1: rel_overloading_1, ..., branch_n: rel_overloading_n},
                [trafo_1, ..., trafo_m]

    Notes:
        Lines'/cables' max. capacity (load case and feed-in case) are taken from [1]_.

    References:
    .. [1] dena VNS

    """

    crit_branches = {}
    crit_stations = []

    if mode == 'MV':
        # load load factors (conditions) for cables, lines and trafos for load- and feedin case
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
        kw2mw = 1e-3

        # 1. check branches' loads
        for branch in grid.graph_edges():
            s_max_th = 3**0.5 * branch['branch'].type['U_n'] * branch['branch'].type['I_max_th']
            # TODO: Add type attribute to branch for checking type !!!!
            if branch['branch'].kind is 'line':
                s_max_th *= load_factor_mv_line_lc_normal
            elif branch['branch'].kind is 'cable':
                s_max_th *= load_factor_mv_cable_lc_normal
            else:
                raise ValueError('Branch kind is invalid!')

            # check loads only for non-aggregated Load Areas (aggregated ones are skipped raising except)
            try:
                if any([s*mw2kw >= s_max_th for s in branch['branch'].s_res]):
                    #crit_branches.append(branch)
                    # save max. relative overloading
                    crit_branches[branch] = max(branch['branch'].s_res) * mw2kw / s_max_th
            except:
                pass

        # TODO: temporarily do not reinforce HV-MV stations

        # # 2. check trafos' loads
        # # get power flow case count
        # # TODO: This way is odd, as soon as there's a central place where PF settings are stored, get it from there
        # pf_case_count = len(branch['branch'].s_res)
        #
        # # max. allowed load of trafo
        # s_max_th_trafo = sum(trafo.s_max_a for trafo in grid._station.transformers())
        #
        # s_max_th_branch = [0] * pf_case_count
        # for node in grid._graph.edge[grid._station]:
        #     branch = grid._graph.edge[grid._station][node]['branch']
        #     if not branch.connects_aggregated:
        #         s_max_th_branch = [sum(_) for _ in zip(s_max_th_branch, branch.s_res)]
        #     else:
        #         # TODO: Currently, peak load is assumed for aggregated LV for all cases!
        #         s_max_th_branch = [sum(_) for _ in zip(s_max_th_branch,
        #                                                pf_case_count * [kw2mw * node.lv_load_area.peak_load_sum])]
        #
        # #print(s_max_th_branch)
        # if any([s*mw2kw >= s_max_th_trafo for s in s_max_th_branch]):
        #     crit_stations.append(grid._station)
        #     # PUT MORE STUFF IN HERE

    elif mode == 'LV':
        raise NotImplementedError

    if crit_branches:
        logger.info('==> {} branches have load issues.'.format(
            len(crit_branches)))
    if crit_stations:
        logger.info('==> {} stations have load issues.'.format(
            len(crit_stations)))

    return crit_branches, crit_stations


def check_voltage(grid, mode):
    """ Checks for voltage stability issues at all nodes for MV or LV grid

    Args:
        grid: GridDingo object
        mode: kind of grid ('MV' or 'LV')

    Returns:
        List of critical nodes, sorted descending by voltage difference

    Notes:
        The examination is done in two steps, according to [1]_:
        1. It is checked

    References:
    .. [1] dena VNS
    """

    crit_nodes = {}

    if mode == 'MV':
        # load max. voltage difference
        mv_max_v_level_diff_normal = float(cfg_dingo.get('mv_routing_tech_constraints',
                                                         'mv_max_v_level_diff_normal'))

        # 1. check nodes' voltages
        voltage_station = grid._station.voltage_res
        for node in grid.graph_nodes_sorted():
            try:
                # compare node's voltage with max. allowed voltage difference
                if any([(v1/v2 > (1 + mv_max_v_level_diff_normal)) or
                        (v1/v2 < (1 - mv_max_v_level_diff_normal))
                        for v1, v2 in zip(node.voltage_res, voltage_station)]):
                    crit_nodes[node] = {'node': node,
                                        'v_diff': max([(v1/v2) for v1, v2 in zip(node.voltage_res, voltage_station)])}
            except:
                pass

    elif mode == 'LV':
        raise NotImplementedError

    if crit_nodes:
        logger.info('==> {} nodes have voltage issues.'.format(len(crit_nodes)))

    return [_['node'] for _ in sorted(crit_nodes.values(), key=lambda _: _['v_diff'], reverse=True)]
