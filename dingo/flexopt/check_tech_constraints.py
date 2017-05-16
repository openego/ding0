# check technical constraints of distribution grids (shared lib)

from dingo.tools import config as cfg_dingo
import logging


logger = logging.getLogger('dingo')


def check_load(grid, mode):
    """ Checks for over-loading of branches and transformers for MV or LV grid

    Parameters
    ----------
    grid: GridDingo object
    mode: String
        kind of grid ('MV' or 'LV')

    Returns
    -------
    Dict of critical branches with max. relative overloading
    List of critical transformers,
    Format: {branch_1: rel_overloading_1, ..., branch_n: rel_overloading_n},
            [trafo_1, ..., trafo_m]

    Notes
    -----
    Lines'/cables' max. capacity (load case and feed-in case) are taken from [1]_.

    References
    ----------
    .. [1] dena VNS

    """

    crit_branches = {}
    crit_stations = []

    if mode == 'MV':
        # load load factors (conditions) for cables, lines and trafos for load- and feedin case

        # load_factor_mv_trans_lc_normal = float(cfg_dingo.get('assumptions',
        #                                                      'load_factor_mv_trans_lc_normal'))
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

        # STEP 1: check branches' loads
        for branch in grid.graph_edges():
            s_max_th = 3**0.5 * branch['branch'].type['U_n'] * branch['branch'].type['I_max_th']
            # TODO: Check LOAD FACTOR!
            if branch['branch'].kind is 'line':
                s_max_th *= load_factor_mv_line_lc_normal
            elif branch['branch'].kind is 'cable':
                s_max_th *= load_factor_mv_cable_lc_normal
            else:
                raise ValueError('Branch kind is invalid!')

            # check loads only for non-aggregated Load Areas (aggregated ones are skipped raising except)
            try:
                if any([s*mw2kw >= s_max_th for s in branch['branch'].s_res]):
                    # save max. relative overloading
                    crit_branches[branch] = max(branch['branch'].s_res) * mw2kw / s_max_th
            except:
                pass

        # STEP 2: check HV-MV station's load

        # NOTE: HV-MV station reinforcement is not required for status-quo
        # scenario since HV-MV trafos already sufficient for load+generation
        # case as done in MVStationDingo.choose_transformers()

        # OLD snippet:
        # cum_peak_load = grid.grid_district.peak_load
        # cum_peak_generation = grid.station().peak_generation(mode='MVLV')
        #
        # # reinforcement necessary only if generation > load
        # if cum_peak_generation > cum_peak_load:
        #     grid.station().choose_transformers
        #
        # cum_trafo_capacity = sum((_.s_max_a for _ in grid.station().transformers()))
        #
        # max_trafo = max((_.s_max_a for _ in grid.station().transformers()))
        #
        # # determine number and size of required transformers
        # kw2mw = 1e-3
        # residual_apparent_power = cum_generation_sum * kw2mw - \
        #                           cum_trafo_capacity

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
