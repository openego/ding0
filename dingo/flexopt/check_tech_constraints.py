# check technical constraints of distribution grids (shared lib)

from dingo.tools import config as cfg_dingo
import logging
from dingo.core.network.loads import LVLoadDingo
from dingo.core.network import GeneratorDingo
import networkx as nx


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
    Dict of critical branches (BranchDingo objects) with max. relative overloading
    List of critical transformers (TransformerDingo objects),
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

    Parameters
    ----------
    grid: GridDingo object
    mode: String
        kind of grid ('MV' or 'LV')

    Returns
    -------
    List of critical nodes, sorted descending by voltage difference

    Notes
    -----
    The voltage is checked against a max. allowed voltage deviation.
    """

    crit_nodes = {}

    if mode == 'MV':
        # load max. voltage difference
        mv_max_v_level_diff_normal = float(cfg_dingo.get('mv_routing_tech_constraints',
                                                         'mv_max_v_level_diff_normal'))

        # check nodes' voltages
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


def assign_line_loading(grid):
    """
    Assign line loading to each branch determined by peak load and peak
    generation of descendant branches

    The attribute `s_res` is a list of two elements
    1. apparent power in load case
    2. apparent power in feed-in case

    Parameters
    ----------
    grid : dingo.core.network.grids.LVGridDingo
        Dingo LV grid object
    """

    # Convert grid to a tree (is a directed graph)
    # based on this tree, descendants of each node are accessible
    # grid_branches = get_branches(grid)

    station = grid._station

    tree = nx.dfs_tree(grid._graph, station)

    for node in tree.nodes():

        # list of descendant nodes including the node itself
        descendants = list(nx.descendants(tree, node))
        descendants.append(node)

        # preceeding node of node
        predecessors = tree.predecessors(node)

        # assing cumulative peak load and generation
        if len(predecessors) > 0:

            # a non-meshed grid topology returns a list with only 1 item
            predecessor = predecessors[0]

            # get preceeding
            branches = grid.graph_branches_from_node(node)
            preceeding_branch = [branch for branch in branches
                                if branch[0] is predecessor][0]

            # determine cumulative peak load at node and assign to branch
            peak_load, peak_gen = peak_load_generation_at_node(descendants)
            preceeding_branch[1]['branch'].s_res = [peak_load, peak_gen]


def peak_load_generation_at_node(nodes):
    """
    Get maximum occuring load and generation at a certain node

    Summarizes peak loads and nominal generation power of descendant nodes
    of a branch

    Parameters
    ----------
    nodes : list
        Any LV grid Dingo node object that is part of the grid topology

    Return
    ------
    peak_load : numeric
        Sum of peak loads of descendant nodes
    peak_generation : numeric
        Sum of nominal power of generation at descendant nodes
    """

    loads = [node.peak_load for node in nodes
             if isinstance(node, LVLoadDingo)]
    peak_load = sum(loads)

    generation = [node.capacity for node in nodes
             if isinstance(node, GeneratorDingo)]
    peak_generation = sum(generation)

    return peak_load, peak_generation