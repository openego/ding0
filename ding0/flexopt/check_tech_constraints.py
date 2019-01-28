"""This file is part of DING0, the DIstribution Network GeneratOr.
DING0 is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DING0 lives at github: https://github.com/openego/ding0/
The documentation is available on RTD: http://ding0.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/ding0/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


# check technical constraints of distribution grids (shared lib)

from ding0.tools import config as cfg_ding0
import logging
from ding0.core.network.loads import LVLoadDing0
from ding0.core.network import GeneratorDing0
from ding0.core.network.cable_distributors import LVCableDistributorDing0
from ding0.core.network.stations import LVStationDing0
import networkx as nx
import math


logger = logging.getLogger('ding0')


def check_load(grid, mode):
    """ Checks for over-loading of branches and transformers for MV or LV grid.

    Parameters
    ----------
    grid : GridDing0
        Grid identifier.
    mode : str
        Kind of grid ('MV' or 'LV').

    Returns
    -------
    :obj:`dict`
        Dict of critical branches with max. relative overloading, and the 
        following format::
        
            {
            branch_1: rel_overloading_1, 
            ..., 
            branch_n: rel_overloading_n
            }
        
    :any:`list` of :obj:`GridDing0`
        List of critical transformers with the following format::
        
        [trafo_1, ..., trafo_m]

    Notes
    -----
        Lines'/cables' max. capacity (load case and feed-in case) are taken from [#]_.
        

    References
    ----------
    .. [#] dena VNS
    
    See Also
    --------
    ding0.flexopt.reinforce_measures.reinforce_branches_current :
    ding0.flexopt.reinforce_measures.reinforce_branches_voltage :
    
    """

    crit_branches = {}
    crit_stations = []

    if mode == 'MV':
        # load load factors (conditions) for cables, lines and trafos for load- and feedin case

        # load_factor_mv_trans_lc_normal = float(cfg_ding0.get('assumptions',
        #                                                      'load_factor_mv_trans_lc_normal'))
        load_factor_mv_line_lc_normal = float(cfg_ding0.get('assumptions',
                                                             'load_factor_mv_line_lc_normal'))
        load_factor_mv_cable_lc_normal = float(cfg_ding0.get('assumptions',
                                                             'load_factor_mv_cable_lc_normal'))
        #load_factor_mv_trans_fc_normal = float(cfg_ding0.get('assumptions',
        #                                                     'load_factor_mv_trans_fc_normal'))
        load_factor_mv_line_fc_normal = float(cfg_ding0.get('assumptions',
                                                             'load_factor_mv_line_fc_normal'))
        load_factor_mv_cable_fc_normal = float(cfg_ding0.get('assumptions',
                                                             'load_factor_mv_cable_fc_normal'))

        mw2kw = 1e3
        kw2mw = 1e-3

        # STEP 1: check branches' loads
        for branch in grid.graph_edges():
            s_max_th = 3**0.5 * branch['branch'].type['U_n'] * branch['branch'].type['I_max_th']

            if branch['branch'].kind == 'line':
                s_max_th_lcfc = [s_max_th * load_factor_mv_line_lc_normal,
                                 s_max_th * load_factor_mv_line_fc_normal]
            elif branch['branch'].kind == 'cable':
                s_max_th_lcfc = [s_max_th * load_factor_mv_cable_lc_normal,
                                 s_max_th * load_factor_mv_cable_fc_normal]
            else:
                raise ValueError('Branch kind is invalid!')

            # check loads only for non-aggregated Load Areas (aggregated ones are skipped raising except)
            try:
                # check if s_res exceeds allowed values for laod and feedin case
                # CAUTION: The order of values is fix! (1. load case, 2. feedin case)
                if any([s_res * mw2kw > _ for s_res, _ in zip(branch['branch'].s_res, s_max_th_lcfc)]):
                    # save max. relative overloading
                    crit_branches[branch] = max(branch['branch'].s_res) * mw2kw / s_max_th
            except:
                pass

        # STEP 2: check HV-MV station's load

        # NOTE: HV-MV station reinforcement is not required for status-quo
        # scenario since HV-MV trafos already sufficient for load+generation
        # case as done in MVStationDing0.choose_transformers()

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
    grid : GridDing0
        Grid identifier.
    mode : str
        Kind of grid ('MV' or 'LV').

    Returns
    -------
    :any:`list` of :any:`GridDing0`
        List of critical nodes, sorted descending by voltage difference.

    Notes
    -----
        The examination is done in two steps, according to [#]_ :
        
        1. It is checked #TODO: what?
        
        2. #TODO: what's next?
        
    References
    ----------
    .. [#] dena VNS
    """

    crit_nodes = {}

    if mode == 'MV':
        # load max. voltage difference for load and feedin case
        mv_max_v_level_lc_diff_normal = float(cfg_ding0.get('mv_routing_tech_constraints',
                                                            'mv_max_v_level_lc_diff_normal'))
        mv_max_v_level_fc_diff_normal = float(cfg_ding0.get('mv_routing_tech_constraints',
                                                            'mv_max_v_level_fc_diff_normal'))

        # check nodes' voltages
        voltage_station = grid._station.voltage_res
        for node in grid.graph_nodes_sorted():
            try:
                # compare node's voltage with max. allowed voltage difference for load and feedin case
                if (abs(voltage_station[0] - node.voltage_res[0]) > mv_max_v_level_lc_diff_normal) or\
                   (abs(voltage_station[1] - node.voltage_res[1]) > mv_max_v_level_fc_diff_normal):

                    crit_nodes[node] = {'node': node,
                                        'v_diff': max([abs(v2-v1) for v1, v2 in zip(node.voltage_res, voltage_station)])}
            except:
                pass

    elif mode == 'LV':
        raise NotImplementedError

    if crit_nodes:
        logger.info('==> {} nodes have voltage issues.'.format(len(crit_nodes)))

    return [_['node'] for _ in sorted(crit_nodes.values(), key=lambda _: _['v_diff'], reverse=True)]


def get_critical_line_loading(grid):
    """
    Assign line loading to each branch determined by peak load and peak
    generation of descendant branches

    The attribute `s_res` is a list of two elements
    1. apparent power in load case
    2. apparent power in feed-in case

    Parameters
    ----------
    grid : ding0.core.network.grids.LVGridDing0
        Ding0 LV grid object

    Returns
    -------
    :any:`list`
        List of critical branches incl. its line loading
    :any:`list`
        List of critical stations incl. its transformer loading
    """
    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    lf_trafo_load = cfg_ding0.get('assumptions',
                                  "load_factor_lv_trans_lc_normal")
    lf_trafo_gen = cfg_ding0.get('assumptions',
                                  "load_factor_lv_trans_fc_normal")

    critical_branches = []
    critical_stations = []

    # Convert grid to a tree (is a directed graph)
    # based on this tree, descendants of each node are accessible
    station = grid._station

    tree = nx.dfs_tree(grid._graph, station)

    for node in tree.nodes():

        # list of descendant nodes including the node itself
        descendants = list(nx.descendants(tree, node))
        descendants.append(node)

        if isinstance(node, LVStationDing0):
            # determine cumulative peak load at node and assign to branch
            peak_load, peak_gen = peak_load_generation_at_node(descendants)

            if grid.id_db == 61107:
                if isinstance(node, LVStationDing0):
                    print(node)
            # get trafos cumulative apparent power
            s_max_trafos = sum([_.s_max_a for _ in node._transformers])

            # compare with load and generation connected to
            if (((peak_load / cos_phi_load) > s_max_trafos * lf_trafo_load) or
                    ((peak_gen / cos_phi_feedin) > s_max_trafos * lf_trafo_gen)):
                critical_stations.append(
                    {'station': node,
                     's_max': [
                         peak_load / cos_phi_load,
                         peak_gen / cos_phi_feedin]})

        else:
            # preceeding node of node
            predecessors = list(tree.predecessors(node))

            # a non-meshed grid topology returns a list with only 1 item
            predecessor = predecessors[0]

            # get preceeding
            branches = grid.graph_branches_from_node(node)
            preceeding_branch = [branch for branch in branches
                                 if branch[0] is predecessor][0]

            # determine cumulative peak load at node and assign to branch
            peak_load, peak_gen = peak_load_generation_at_node(descendants)

            s_max_th = 3 ** 0.5 * preceeding_branch[1]['branch'].type['U_n'] * \
                       preceeding_branch[1]['branch'].type['I_max_th'] / 1e3

            if (((peak_load / cos_phi_load) > s_max_th) or
                    ((peak_gen / cos_phi_feedin) > s_max_th)):
                critical_branches.append(
                    {'branch': preceeding_branch[1]['branch'],
                     's_max': [
                         peak_load / cos_phi_load,
                         peak_gen / cos_phi_feedin]})

    return critical_branches, critical_stations


def peak_load_generation_at_node(nodes):
    """
    Get maximum occuring load and generation at a certain node

    Summarizes peak loads and nominal generation power of descendant nodes
    of a branch

    Parameters
    ----------
    nodes : :any:`list`
        Any LV grid Ding0 node object that is part of the grid topology

    Return
    ------
    :any:`float`
        peak_load : Sum of peak loads of descendant nodes
    :any:`float`
        peak_generation : Sum of nominal power of generation at descendant nodes
    """

    loads = [node.peak_load for node in nodes
             if isinstance(node, LVLoadDing0)]
    peak_load = sum(loads)

    generation = [node.capacity for node in nodes
             if isinstance(node, GeneratorDing0)]
    peak_generation = sum(generation)

    return peak_load, peak_generation


def get_critical_voltage_at_nodes(grid):
    """
    Estimate voltage drop/increase induced by loads/generators connected to the
    grid.

    Based on voltage level at each node of the grid critical nodes in terms
    of exceed tolerable voltage drop/increase are determined.
    The tolerable voltage drop/increase is defined by [#VDE]_ a adds up to
    3 % of nominal voltage.
    The longitudinal voltage drop at each line segment is estimated by a
    simplified approach (neglecting the transverse voltage drop) described in
    [#VDE]_.

    Two equations are available for assessing voltage drop/ voltage increase.

    The first is used to assess a voltage drop in the load case

    .. math::
        \\Delta u = \\frac{S_{Amax} \cdot ( R_{kV} \cdot cos(\phi) + X_{kV} \cdot sin(\phi) )}{U_{nom}}

    The second equation can be used to assess the voltage increase in case of
    feedin. The only difference is the negative sign before X. This is related
    to consider a voltage drop due to inductive operation of generators.

    .. math::
        \\Delta u = \\frac{S_{Amax} \cdot ( R_{kV} \cdot cos(\phi) - X_{kV} \cdot sin(\phi) )}{U_{nom}}

    =================  =============================
    Symbol             Description
    =================  =============================
    :math:`\Delta u`   Voltage drop/increase at node
    :math:`S_{Amax}`   Apparent power
    :math:`R_{kV}`     Short-circuit resistance
    :math:`X_{kV}`     Short-circuit reactance
    :math:`cos(\phi)`  Power factor
    :math:`U_{nom}`    Nominal voltage
    =================  =============================

    Parameters
    ----------
    grid : LVGridDing0
        Ding0 LV grid object

    Notes
    -----
    The implementation highly depends on topology of LV grid. This must not
    change its topology from radial grid with stubs branching from radial
    branches. In general, the approach of [#VDE]_ is only applicable to grids of
    radial topology.

    We consider the transverse voltage drop/increase by applying the same
    methodology successively on results of main branch. The voltage
    drop/increase at each house connection branch (aka. stub branch or grid
    connection point) is estimated by superposition based on voltage level
    in the main branch cable distributor.
    
    References
    ----------
    .. [#VDE] VDE Anwenderrichtlinie: Erzeugungsanlagen am Niederspannungsnetz –
        Technische Mindestanforderungen für Anschluss und Parallelbetrieb von
        Erzeugungsanlagen am Niederspannungsnetz, 2011
    """

    v_delta_tolerable_fc = cfg_ding0.get('assumptions',
                                      'lv_max_v_level_fc_diff_normal')
    v_delta_tolerable_lc = cfg_ding0.get('assumptions',
                                      'lv_max_v_level_lc_diff_normal')

    omega = 2 * math.pi * 50

    crit_nodes = []

    # get list of nodes of main branch in right order
    tree = nx.dfs_tree(grid._graph, grid._station)

    # list for nodes of main branch
    main_branch = []

    # list of stub cable distributors branching from main branch
    grid_conn_points = []

    # fill two above lists
    for node in list(nx.descendants(tree, grid._station)):
        successors = list(tree.successors(node))
        if successors and all(isinstance(successor, LVCableDistributorDing0)
               for successor in successors):
            main_branch.append(node)
        elif (isinstance(node, LVCableDistributorDing0) and
            all(isinstance(successor, (GeneratorDing0, LVLoadDing0))
               for successor in successors)):
            grid_conn_points.append(node)

    # voltage at substation bus bar
    r_mv_grid, x_mv_grid = get_mv_impedance(grid)

    r_trafo = sum([tr.r for tr in grid._station._transformers])
    x_trafo = sum([tr.x for tr in grid._station._transformers])

    v_delta_load_case_bus_bar, \
    v_delta_gen_case_bus_bar = get_voltage_at_bus_bar(grid, tree)

    if (abs(v_delta_gen_case_bus_bar) > v_delta_tolerable_fc
        or abs(v_delta_load_case_bus_bar) > v_delta_tolerable_lc):
        crit_nodes.append({'node': grid._station,
                           'v_diff': [v_delta_load_case_bus_bar,
                                      v_delta_gen_case_bus_bar]})



    # voltage at main route nodes
    for first_node in [b for b in tree.successors(grid._station)
                   if b in main_branch]:

        # cumulative resistance/reactance at bus bar
        r = r_mv_grid + r_trafo
        x = x_mv_grid + x_trafo

        # roughly estimate transverse voltage drop
        stub_node = [_ for _ in list(tree.successors(first_node)) if
                     _ not in main_branch][0]
        v_delta_load_stub, v_delta_gen_stub = voltage_delta_stub(
            grid,
            tree,
            first_node,
            stub_node,
            r,
            x)

        # cumulative voltage drop/increase at substation bus bar
        v_delta_load_cum = v_delta_load_case_bus_bar
        v_delta_gen_cum = v_delta_gen_case_bus_bar

        # calculate voltage at first node of branch
        voltage_delta_load, voltage_delta_gen, r, x = \
            get_voltage_delta_branch(grid, tree, first_node, r, x)

        v_delta_load_cum += voltage_delta_load
        v_delta_gen_cum += voltage_delta_gen

        if (abs(v_delta_gen_cum) > (v_delta_tolerable_fc - v_delta_gen_stub)
            or abs(v_delta_load_cum) > (v_delta_tolerable_lc - v_delta_load_stub)):
            crit_nodes.append({'node': first_node,
                               'v_diff': [v_delta_load_cum,
                                          v_delta_gen_cum]})
            crit_nodes.append({'node': stub_node,
                               'v_diff': [
                                   v_delta_load_cum + v_delta_load_stub,
                                   v_delta_gen_cum + v_delta_gen_stub]})

        # get next neighboring nodes down the tree
        successor = [x for x in tree.successors(first_node)
                      if x in main_branch]
        if successor:
            successor = successor[0] # simply unpack

        # successively determine voltage levels for succeeding nodes
        while successor:
            voltage_delta_load, voltage_delta_gen, r, x = \
                get_voltage_delta_branch(grid, tree, successor, r, x)

            v_delta_load_cum += voltage_delta_load
            v_delta_gen_cum += voltage_delta_gen

            # roughly estimate transverse voltage drop
            stub_node = [_ for _ in tree.successors(successor) if
                         _ not in main_branch][0]
            v_delta_load_stub, v_delta_gen_stub = voltage_delta_stub(
                grid,
                tree,
                successor,
                stub_node,
                r,
                x)

            if (abs(v_delta_gen_cum) > (v_delta_tolerable_fc - v_delta_gen_stub)
                or abs(v_delta_load_cum) > (
                            v_delta_tolerable_lc - v_delta_load_stub)):
                crit_nodes.append({'node': successor,
                                   'v_diff': [v_delta_load_cum,
                                              v_delta_gen_cum]})
                crit_nodes.append({'node': stub_node,
                                   'v_diff': [
                                       v_delta_load_cum + v_delta_load_stub,
                                       v_delta_gen_cum + v_delta_gen_stub]})

            successor = [_ for _ in tree.successors(successor)
                         if _ in main_branch]
            if successor:
                successor = successor[0]

    return crit_nodes


def voltage_delta_vde(v_nom, s_max, r, x, cos_phi):
    """
    Estimate voltrage drop/increase

    The VDE [#]_ proposes a simplified method to estimate voltage drop or
    increase in radial grids.

    Parameters
    ----------
    v_nom : int
        Nominal voltage
    s_max : float
        Apparent power
    r : float
        Short-circuit resistance from node to HV/MV substation (in ohm)
    x : float
        Short-circuit reactance from node to HV/MV substation (in ohm). Must
        be a signed number indicating (+) inductive reactive consumer (load
        case) or (-) inductive reactive supplier (generation case)
    cos_phi : float

    Returns
    -------
    :any:`float`
        Voltage drop or increase
        
    References
    ----------
    .. [#] VDE Anwenderrichtlinie: Erzeugungsanlagen am Niederspannungsnetz –
        Technische Mindestanforderungen für Anschluss und Parallelbetrieb von
        Erzeugungsanlagen am Niederspannungsnetz, 2011

    """
    delta_v = (s_max * (
        r * cos_phi + x * math.sin(math.acos(cos_phi)))) / v_nom ** 2
    return delta_v


def get_house_conn_gen_load(graph, node):
    """
    Get generation capacity/ peak load of neighboring house connected to main
    branch

    Parameters
    ----------
    graph : :networkx:`NetworkX Graph Obj< >`
        Directed graph
    node : graph node
        Node of the main branch of LV grid

    Returns
    -------
    :any:`list`
        A list containing two items
        
        # peak load of connected house branch
        # generation capacity of connected generators
    """
    generation = 0
    peak_load = 0

    for cus_1 in graph.successors(node):
        for cus_2 in graph.successors(cus_1):
            if not isinstance(cus_2, list):
                cus_2 = [cus_2]
            generation += sum([gen.capacity for gen in cus_2
                          if isinstance(gen, GeneratorDing0)])
            peak_load += sum([load.peak_load for load in cus_2
                          if isinstance(load, LVLoadDing0)])

    return [peak_load, generation]


def get_voltage_delta_branch(grid, tree, node, r_preceeding, x_preceeding):
    """
    Determine voltage for a preceeding branch (edge) of node

    Parameters
    ----------
    grid : LVGridDing0
        Ding0 grid object
    tree : :networkx:`NetworkX Graph Obj< >`
        Tree of grid topology
    node : graph node
        Node to determine voltage level at
    r_preceeding : float
        Resitance of preceeding grid
    x_preceeding : float
        Reactance of preceeding grid

    Return
    ------
    :any:`float`
        Delta voltage for node
    """
    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage')
    omega = 2 * math.pi * 50

    # add resitance/ reactance to preceeding
    in_edge = [_ for _ in grid.graph_branches_from_node(node) if
               _[0] in list(tree.predecessors(node))][0][1]
    r = r_preceeding + (in_edge['branch'].type['R'] *
                     in_edge['branch'].length)
    x = x_preceeding + (in_edge['branch'].type['L'] / 1e3 * omega *
                     in_edge['branch'].length)

    # get apparent power for load and generation case
    peak_load, gen_capacity = get_house_conn_gen_load(tree, node)
    s_max_load = peak_load / cos_phi_load
    s_max_feedin = gen_capacity / cos_phi_feedin

    # determine voltage increase/ drop a node
    voltage_delta_load = voltage_delta_vde(v_nom, s_max_load, r, x,
                                           cos_phi_load)
    voltage_delta_gen = voltage_delta_vde(v_nom, s_max_feedin, r, -x,
                                          cos_phi_feedin)

    return [voltage_delta_load, voltage_delta_gen, r, x]


def get_mv_impedance(grid):
    """
    Determine MV grid impedance (resistance and reactance separately)

    Parameters
    ----------
    grid : LVGridDing0

    Returns
    -------
    :any:`list`
        List containing resistance and reactance of MV grid
    """

    omega = 2 * math.pi * 50

    mv_grid = grid.grid_district.lv_load_area.mv_grid_district.mv_grid
    edges = mv_grid.find_path(grid._station, mv_grid._station, type='edges')
    r_mv_grid = sum([e[2]['branch'].type['R'] * e[2]['branch'].length / 1e3
                     for e in edges])
    x_mv_grid = sum([e[2]['branch'].type['L'] / 1e3 * omega * e[2][
        'branch'].length / 1e3 for e in edges])

    return [r_mv_grid, x_mv_grid]


def voltage_delta_stub(grid, tree, main_branch_node, stub_node, r_preceeding,
                       x_preceedig):
    """
    Determine voltage for stub branches

    Parameters
    ----------
    grid : LVGridDing0
        Ding0 grid object
    tree : :networkx:`NetworkX Graph Obj< >`
        Tree of grid topology
    main_branch_node : graph node
        Node of main branch that stub branch node in connected to
    main_branch : dict
        Nodes of main branch
    r_preceeding : float
        Resitance of preceeding grid
    x_preceeding : float
        Reactance of preceeding grid

    Return
    ------
    :any:`float`
        Delta voltage for node
    """
    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage')
    omega = 2 * math.pi * 50

    stub_branch = [_ for _ in grid.graph_branches_from_node(main_branch_node) if
                   _[0] == stub_node][0][1]
    r_stub = stub_branch['branch'].type['R'] * stub_branch[
        'branch'].length / 1e3
    x_stub = stub_branch['branch'].type['L'] / 1e3 * omega * \
             stub_branch['branch'].length / 1e3
    s_max_gen = [_.capacity / cos_phi_feedin
                 for _ in tree.successors(stub_node)
                 if isinstance(_, GeneratorDing0)]
    if s_max_gen:
        s_max_gen = s_max_gen[0]
        v_delta_stub_gen = voltage_delta_vde(v_nom, s_max_gen, r_stub + r_preceeding,
                                             x_stub + x_preceedig, cos_phi_feedin)
    else:
        v_delta_stub_gen = 0

    s_max_load = [_.peak_load / cos_phi_load
                  for _ in tree.successors(stub_node)
                  if isinstance(_, LVLoadDing0)]
    if s_max_load:
        s_max_load = s_max_load[0]
        v_delta_stub_load = voltage_delta_vde(v_nom, s_max_load, r_stub + r_preceeding,
                                              x_stub + x_preceedig, cos_phi_load)
    else:
        v_delta_stub_load = 0

    return [v_delta_stub_load, v_delta_stub_gen]


def get_voltage_at_bus_bar(grid, tree):
    """
    Determine voltage level at bus bar of MV-LV substation

    Parameters
    ----------
    grid : LVGridDing0
        Ding0 grid object
    tree : :networkx:`NetworkX Graph Obj< >`
        Tree of grid topology:

    Returns
    -------
    :any:`list`
        Voltage at bus bar. First item refers to load case, second item refers
        to voltage in feedin (generation) case
    """

    # voltage at substation bus bar
    r_mv_grid, x_mv_grid = get_mv_impedance(grid)

    r_trafo = sum([tr.r for tr in grid._station._transformers])
    x_trafo = sum([tr.x for tr in grid._station._transformers])

    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage')

    # loads and generators connected to bus bar
    bus_bar_load = sum(
        [node.peak_load for node in tree.successors(grid._station)
         if isinstance(node, LVLoadDing0)]) / cos_phi_load
    bus_bar_generation = sum(
        [node.capacity for node in tree.successors(grid._station)
         if isinstance(node, GeneratorDing0)]) / cos_phi_feedin

    v_delta_load_case_bus_bar = voltage_delta_vde(v_nom,
                                                  bus_bar_load,
                                                  (r_mv_grid + r_trafo),
                                                  (x_mv_grid + x_trafo),
                                                  cos_phi_load)
    v_delta_gen_case_bus_bar = voltage_delta_vde(v_nom,
                                                 bus_bar_generation,
                                                 (r_mv_grid + r_trafo),
                                                 -(x_mv_grid + x_trafo),
                                                 cos_phi_feedin)

    return v_delta_load_case_bus_bar, v_delta_gen_case_bus_bar
