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
from ding0.core.powerflow import q_sign
import networkx as nx
import math


logger = logging.getLogger('ding0')


def check_load(grid, mode):
    """ Checks for over-loading of branches and transformers for MV or LV grid.

    Parameters
    ----------
    grid : :class:`~.ding0.core.GridDing0`
        Grid identifier.
    mode : :obj:`str`
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
        
    :obj:`list` of :class:`~.ding0.core.network.TransformerDing0` objects
        List of critical transformers with the following format::
        
        [trafo_1, ..., trafo_m]

    Note
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
                    #crit_branches[branch] = max(branch['branch'].s_res) * mw2kw / s_max_th
                    crit_branches[branch['branch']] = max(branch['branch'].s_res) * mw2kw / s_max_th # PAUL new made change, this fct did not work
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
    grid : :class:`~.ding0.core.GridDing0`
        Grid identifier.
    mode : :obj:`str`
        Kind of grid ('MV' or 'LV').

    Returns
    -------
    :obj:`list` of Ding0 node object (member of graph) either

        * :class:`~.ding0.core.network.GeneratorDing0` or
        * :class:`~.ding0.core.network.GeneratorFluctuatingDing0` or
        * :class:`~.ding0.core.network.LoadDing0` or
        * :class:`~.ding0.core.network.StationDing0` or
        * :class:`~.ding0.core.network.CircuitBreakerDing0` or
        * :class:`~.ding0.core.network.CableDistributorDing0`

        List of critical nodes, sorted descending by voltage difference.

    Note
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
    grid : :class:`~.ding0.core.network.grids.LVGridDing0`
        Ding0 LV grid object

    Returns
    -------
    :obj:`list`
        List of critical branches incl. its line loading
    :obj:`list`
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

    tree = nx.dfs_tree(grid.graph, station)

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
    nodes : :obj:`list`
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
    r"""
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
    grid : :class:`~.ding0.core.network.grids.LVGridDing0`
        Ding0 LV grid object

    Note
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

    crit_nodes = []

    # get list of nodes of main branch in right order
    tree = nx.dfs_tree(grid.graph, grid._station)

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

    v_delta_load_case_bus_bar, v_delta_gen_case_bus_bar  = get_voltage_at_bus_bar(grid, tree)

    if (abs(v_delta_gen_case_bus_bar) > v_delta_tolerable_fc
        or abs(v_delta_load_case_bus_bar) > v_delta_tolerable_lc):
        crit_nodes.append({'node': grid._station,
                           'v_diff': [v_delta_load_case_bus_bar,
                                      v_delta_gen_case_bus_bar]})



    # voltage at main route nodes
    for first_node in [b for b in tree.successors(grid._station)
                   if b in main_branch]:

        # initiate loop over feeder
        successor = first_node
        # cumulative voltage drop/increase at substation bus bar
        v_delta_load_cum = v_delta_load_case_bus_bar
        v_delta_gen_cum = v_delta_gen_case_bus_bar

        # successively determine voltage levels for succeeding nodes
        while successor:
            # calculate voltage drop over preceding line
            voltage_delta_load, voltage_delta_gen  = get_delta_voltage_preceding_line(grid, tree, successor)
            # add voltage drop over preceding line
            v_delta_load_cum += voltage_delta_load
            v_delta_gen_cum += voltage_delta_gen

            # roughly estimate transverse voltage drop
            stub_node = [_ for _ in tree.successors(successor) if
                         _ not in main_branch][0]
            v_delta_load_stub, v_delta_gen_stub  = get_delta_voltage_preceding_line(grid, tree, stub_node)

            # check if voltage drop at node exceeds tolerable voltage drop
            if (abs(v_delta_gen_cum) > (v_delta_tolerable_fc)
                or abs(v_delta_load_cum) > (
                            v_delta_tolerable_lc)):
                # add node and successing stub node to critical nodes
                crit_nodes.append({'node': successor,
                                   'v_diff': [v_delta_load_cum,
                                              v_delta_gen_cum]})
                crit_nodes.append({'node': stub_node,
                                   'v_diff': [
                                       v_delta_load_cum + v_delta_load_stub,
                                       v_delta_gen_cum + v_delta_gen_stub]})
            # check if voltage drop at stub node exceeds tolerable voltage drop
            elif ((abs(v_delta_gen_cum + v_delta_gen_stub) > v_delta_tolerable_fc)
                or (abs(v_delta_load_cum + v_delta_load_stub) > v_delta_tolerable_lc)):
                # add stub node to critical nodes
                crit_nodes.append({'node': stub_node,
                                   'v_diff': [
                                       v_delta_load_cum + v_delta_load_stub,
                                       v_delta_gen_cum + v_delta_gen_stub]})


            successor = [_ for _ in tree.successors(successor)
                         if _ in main_branch]
            if successor:
                successor = successor[0]

    return crit_nodes


def get_voltage_at_bus_bar(grid, tree):
    """
        Determine voltage level at bus bar of MV-LV substation

        Parameters
        ----------
        grid : :class:`~.ding0.core.network.grids.LVGridDing0`
            Ding0 grid object
        tree : :networkx:`NetworkX Graph Obj< >`
            Tree of grid topology:

        Returns
        -------
        :obj:`list`
            Voltage at bus bar. First item refers to load case, second item refers
            to voltage in feedin (generation) case
        """
    # impedance of mv grid and transformer
    r_mv_grid, x_mv_grid = get_mv_impedance_at_voltage_level(grid, grid.v_level / 1e3)
    z_trafo = 1 / sum(1 / (tr.z(voltage_level=grid.v_level / 1e3)) for tr in grid._station._transformers)
    r_trafo = z_trafo.real
    x_trafo = z_trafo.imag
    # cumulative resistance/reactance at bus bar
    r_busbar = r_mv_grid + r_trafo
    x_busbar = x_mv_grid + x_trafo
    # get voltage drop at substation bus bar
    v_delta_load_case_bus_bar, \
    v_delta_gen_case_bus_bar = get_voltage_delta_branch(tree, grid._station, r_busbar, x_busbar)
    return v_delta_load_case_bus_bar, v_delta_gen_case_bus_bar


def get_delta_voltage_preceding_line(grid, tree, node):
    """
    Parameters
    ----------
    grid : :class:`~.ding0.core.network.grids.LVGridDing0`
        Ding0 grid object
    tree: :networkx:`NetworkX Graph Obj< >`
        Tree of grid topology
    node: graph node
        Node at end of line
    Return
    ------
    :any:`float`
        Voltage drop over preceding line of node
    """

    # get impedance of preceding line
    freq = cfg_ding0.get('assumptions', 'frequency')
    omega = 2 * math.pi * freq

    # choose preceding branch
    branch = [_ for _ in grid.graph_branches_from_node(node) if
              _[0] in list(tree.predecessors(node))][0][1]

    # calculate impedance of preceding branch
    r_line = (branch['branch'].type['R_per_km'] * branch['branch'].length/1e3)
    x_line = (branch['branch'].type['L_per_km'] / 1e3 * omega *
         branch['branch'].length/1e3)

    # get voltage drop over preceeding line
    voltage_delta_load, voltage_delta_gen = \
        get_voltage_delta_branch(tree, node, r_line, x_line)

    return voltage_delta_load, voltage_delta_gen


def get_voltage_delta_branch(tree, node, r, x):
    """
    Determine voltage for a branch with impedance r + jx

    Parameters
    ----------
    tree : :networkx:`NetworkX Graph Obj< >`
        Tree of grid topology
    node : graph node
        Node to determine voltage level at
    r : float
        Resistance of preceeding branch
    x : float
        Reactance of preceeding branch

    Return
    ------
    :any:`float`
        Delta voltage for branch
    """
    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    cos_phi_load_mode = cfg_ding0.get('assumptions', 'cos_phi_load_mode')
    cos_phi_feedin_mode = cfg_ding0.get('assumptions', 'cos_phi_gen_mode') #ToDo: Check if this is true. Why would generator run in a way that aggravates voltage issues?
    v_nom = cfg_ding0.get('assumptions', 'lv_nominal_voltage')

    # get apparent power for load and generation case
    peak_load, gen_capacity = get_cumulated_conn_gen_load(tree, node)
    s_max_load = peak_load/cos_phi_load
    s_max_feedin = gen_capacity/cos_phi_feedin

    # determine voltage increase/ drop a node
    x_sign_load = q_sign(cos_phi_load_mode, 'load')
    voltage_delta_load = voltage_delta_vde(v_nom, s_max_load, r, x_sign_load * x,
                                           cos_phi_load)
    x_sign_gen = q_sign(cos_phi_feedin_mode, 'load')
    voltage_delta_gen = voltage_delta_vde(v_nom, s_max_feedin, r, x_sign_gen * x,
                                          cos_phi_feedin)

    return [voltage_delta_load, voltage_delta_gen]


def get_cumulated_conn_gen_load(graph, node):
    """
    Get generation capacity/ peak load of all descending nodes

    Parameters
    ----------
    graph : :networkx:`NetworkX Graph Obj< >`
        Directed graph
    node : graph node
        Node of the main branch of LV grid

    Returns
    -------
    :obj:`list`
        A list containing two items

        # cumulated peak load of connected loads at descending nodes of node
        # cumulated generation capacity of connected generators at descending nodes of node
    """

    # loads and generators connected to descending nodes
    peak_load = sum(
        [node.peak_load for node in nx.descendants(graph, node)
         if isinstance(node, LVLoadDing0)])
    generation = sum(
        [node.capacity for node in nx.descendants(graph, node)
         if isinstance(node, GeneratorDing0)])
    return [peak_load, generation]


def get_mv_impedance_at_voltage_level(grid, voltage_level):
    """
    Determine MV grid impedance (resistance and reactance separately)

    Parameters
    ----------
    grid : :class:`~.ding0.core.network.grids.LVGridDing0`
    voltage_level: float
        voltage level to which impedance is rescaled (normally 0.4 kV for LV)

    Returns
    -------
    :obj:`list`
        List containing resistance and reactance of MV grid
    """

    freq = cfg_ding0.get('assumptions', 'frequency')
    omega = 2 * math.pi * freq

    mv_grid = grid.grid_district.lv_load_area.mv_grid_district.mv_grid
    edges = mv_grid.find_path(grid._station, mv_grid._station, type='edges')
    r_mv_grid = sum([e[2]['branch'].type['R_per_km'] * e[2]['branch'].length / 1e3
                     for e in edges])
    x_mv_grid = sum([e[2]['branch'].type['L_per_km'] / 1e3 * omega * e[2][
        'branch'].length / 1e3 for e in edges])
    # rescale to voltage level
    r_mv_grid_vl = r_mv_grid * (voltage_level / mv_grid.v_level) ** 2
    x_mv_grid_vl = x_mv_grid * (voltage_level / mv_grid.v_level) ** 2
    return [r_mv_grid_vl, x_mv_grid_vl]


def voltage_delta_vde(v_nom, s_max, r, x, cos_phi):
    """
    Estimate voltrage drop/increase

    The VDE [#]_ proposes a simplified method to estimate voltage drop or
    increase in radial grids.

    Parameters
    ----------
    v_nom : :obj:`int`
        Nominal voltage
    s_max : :obj:`float`
        Apparent power
    r : :obj:`float`
        Short-circuit resistance from node to HV/MV substation (in ohm)
    x : :obj:`float`
        Short-circuit reactance from node to HV/MV substation (in ohm). Must
        be a signed number indicating (+) inductive reactive consumer (load
        case) or (-) inductive reactive supplier (generation case)
    cos_phi : :obj:`float`
        The cosine phi of the connected generator or load that induces the
        voltage change

    Returns
    -------
    :obj:`float`
        Voltage drop or increase

    References
    ----------
    .. [#] VDE Anwenderrichtlinie: Erzeugungsanlagen am Niederspannungsnetz –
        Technische Mindestanforderungen für Anschluss und Parallelbetrieb von
        Erzeugungsanlagen am Niederspannungsnetz, 2011

    """
    delta_v = (s_max * 1e3 * (
            r * cos_phi - x * math.sin(math.acos(cos_phi)))) / v_nom ** 2
    return delta_v
