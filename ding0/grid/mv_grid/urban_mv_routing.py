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


import time

from ding0.grid.mv_grid.models.models import Graph, Node
from ding0.grid.mv_grid.util import util, data_input
from ding0.grid.mv_grid.solvers import savings, local_search
from ding0.tools.geo import calc_geo_dist, calc_geo_dist_matrix, calc_geo_centre_point, calc_edge_geometry
from ding0.tools import config as cfg_ding0
from ding0.core.network.stations import *
from ding0.core.structure.regions import LVLoadAreaCentreDing0
from ding0.core.network import RingDing0, BranchDing0, CircuitBreakerDing0
from ding0.core.network.cable_distributors import MVCableDistributorDing0
import logging

from ding0.grid.mv_grid.tools import get_edge_tuples_from_path, cut_line_by_distance, reduce_graph_for_dist_matrix_calc, \
calc_street_dist_matrix, conn_ding0_obj_to_osm_graph, get_shortest_path_shp_single_target, \
update_graphs, create_stub_dict, check_stub_criterion, update_stub_dict, split_graph_by_core, relabel_graph_nodes
from shapely.ops import linemerge
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString
#from ding0.tools.plots import plot_mv_topology
from ding0.grid.mv_grid.urban_mv_connect import mv_urban_connect

logger = logging.getLogger(__name__)

def osm_graph_to_routing_specs_urban(agg_load_area, core_graph, depot_node, nodes_pos, nodes_demands):
    
    # agg_load_area, core_graph, depot_node, nodes_pos, nodes_demands 

    G = core_graph

    specs = {} 

    nodes_demands = {str(k):v for k,v in nodes_demands.items() if str(k) in core_graph.nodes}
    nodes_pos = {str(k):v for k,v in nodes_pos.items() if str(k) in core_graph.nodes}

    #depot node
    nodes_demands[str(depot_node)] = 0
    nodes_pos[str(depot_node)] = (depot_node.geo_data.x, depot_node.geo_data.y)

    # fill specs for routing
    specs['DEPOT'] = str(depot_node)
    specs['BRANCH_KIND'] = agg_load_area.mv_grid_district.mv_grid.default_branch_kind
    specs['BRANCH_TYPE'] = agg_load_area.mv_grid_district.mv_grid.default_branch_type
    specs['V_LEVEL'] = agg_load_area.mv_grid_district.mv_grid.v_level
    specs['NODE_COORD_SECTION'] = nodes_pos
    specs['DEMAND'] = nodes_demands
    # nodes_agg excludes node from ring routing, therefore set to False
    specs['IS_AGGREGATED'] = {k: False for k,v in nodes_demands.items() if k in core_graph.nodes}

    # compute distance street matrix
    matrix_node_list = list(nodes_demands.keys())
    specs['MATRIX'] = calc_street_dist_matrix(core_graph, matrix_node_list)
    
    return specs


def routing_solution_to_ding0_graph(mv_grid, core_graph, solution):
    #mv_grid
    """ Insert `solution` from routing into `graph`

    Parameters
    ----------
    graph: :networkx:`NetworkX Graph Obj< >`
    NetworkX graph object with nodes
    solution: BaseSolution
    Instance of `BaseSolution` or child class (e.g. `LocalSearchSolution`) (=solution from routing)

    Returns
    -------
    :networkx:`NetworkX Graph Obj< >` 
    NetworkX graph object with nodes and edges
    """

    osmid_branch_dict = {}

    # build node dict (name: obj) from graph nodes to map node names on node objects
    node_list = {str(n): n for n in mv_grid.graph.nodes()}

    # add edges from solution to graph
    try:
        depot = solution._nodes[solution._problem._depot.name()]
        depot_node = node_list[depot.name()]

        for r in solution.routes():
            circ_breaker_pos = None

            # if route has only one node and is not aggregated, it wouldn't be possible to add two lines from and to
            # this node (undirected graph of NetworkX). So, as workaround, an additional MV cable distributor is added
            # at nodes' position (resulting route: HV/MV_subst --- node --- cable_dist --- HV/MV_subst.
            if len(r._nodes) == 1:
                if not solution._problem._is_aggregated[r._nodes[0]._name]:

                    node = node_list[r._nodes[0]._name]
                    # create new cable dist
                    cable_dist = MVCableDistributorDing0(geo_data=node.geo_data,
                                                         grid=mv_grid)
                    mv_grid.add_cable_distributor(cable_dist)
                    #add cable_dist to osm_graph
                    core_graph.add_node(str(cable_dist), geometry=node.geo_data)
                    core_graph.add_edge(str(cable_dist), str(node), 0, geometry=LineString([node.geo_data, node.geo_data]))
                    core_graph.add_edge(str(node), str(cable_dist), 0, geometry=LineString([node.geo_data, node.geo_data]))

                    # create new node (as dummy) an allocate to route r
                    r.allocate([Node(name=repr(cable_dist), demand=0)])

                    # add it to node list and allocated-list manually
                    node_list[str(cable_dist)] = cable_dist
                    solution._problem._is_aggregated[str(cable_dist)] = False

                    # set circ breaker pos manually
                    circ_breaker_pos = 1

            # build edge list
            n1 = r._nodes[0:len(r._nodes)-1]
            n2 = r._nodes[1:len(r._nodes)]
            edges = list(zip(n1, n2))
            edges.insert(0, (depot, r._nodes[0]))
            edges.append((r._nodes[-1], depot))

            # create MV Branch object for every edge in `edges`
            mv_branches = [BranchDing0(grid=mv_grid) for _ in edges]
            edges_with_branches = list(zip(edges, mv_branches))

            # recalculate circuit breaker positions for final solution, create it and set associated branch.
            # if circ. breaker position is not set manually (routes with more than one load area, see above)
            if not circ_breaker_pos:
                circ_breaker_pos = r.calc_circuit_breaker_position()

            node1 = node_list[edges[circ_breaker_pos - 1][0].name()]
            node2 = node_list[edges[circ_breaker_pos - 1][1].name()]

            # ALTERNATIVE TO METHOD ABOVE: DO NOT CREATE 2 BRANCHES (NO RING) -> LA IS CONNECTED AS SATELLITE
            # IF THIS IS COMMENTED-IN, THE IF-BLOCK IN LINE 87 HAS TO BE COMMENTED-OUT
            # See issue #114
            # ===============================
            # do not add circuit breaker for routes which are aggregated load areas or
            # routes that contain only one load area
            # if not (node1 == depot_node and solution._problem._is_aggregated[edges[circ_breaker_pos - 1][1].name()] or
            #         node2 == depot_node and solution._problem._is_aggregated[edges[circ_breaker_pos - 1][0].name()] or
            #         len(r._nodes) == 1):
            # ===============================

            # do not add circuit breaker for routes which are aggregated load areas
            if not (node1 == depot_node and solution._problem._is_aggregated[edges[circ_breaker_pos - 1][1].name()] or
                    node2 == depot_node and solution._problem._is_aggregated[edges[circ_breaker_pos - 1][0].name()]):
                branch = mv_branches[circ_breaker_pos - 1]

                line_shp = get_shortest_path_shp_single_target(core_graph, node1, node2)[0]

                circ_breaker = CircuitBreakerDing0(grid=mv_grid, branch=branch,
                                                   geo_data=cut_line_by_distance(line_shp, 0.5, normalized=True)[0])
                branch.circuit_breaker = circ_breaker

            # create new ring object for route
            ring = RingDing0(grid=mv_grid)

            # ring demand is apparent power [kVA]
            ring._demand = sum([node.demand() for node in r._nodes])

            # translate solution's node names to graph node objects using dict created before
            # note: branch object is assigned to edge using an attribute ('branch' is used here), it can be accessed
            # using the method `graph_edges()` of class `GridDing0`
            edges_graph = []
            for ((n1, n2), b) in edges_with_branches:
                # get node objects
                node1 = node_list[n1.name()]
                node2 = node_list[n2.name()]

                # set branch's ring attribute
                b.ring = ring

                # set LVLA's ring attribute
                if isinstance(node1, LVLoadAreaCentreDing0):
                    node1.lv_load_area.ring = ring

                # calculate branch geometry and length for street courses
                line_shp, line_length, sp = get_shortest_path_shp_single_target(core_graph, node1, node2, return_path=True)

                b.geometry = line_shp
                b.length = line_length

                # set branch kind and type
                # 1) default
                b.kind = mv_grid.default_branch_kind
                b.type = mv_grid.default_branch_type
                # 2) aggregated load area types
                if node1 == depot_node and solution._problem._is_aggregated[n2.name()]:
                    b.connects_aggregated = True
                    b.kind = mv_grid.default_branch_kind_aggregated
                    b.type = mv_grid.default_branch_type_aggregated
                elif node2 == depot_node and solution._problem._is_aggregated[n1.name()]:
                    b.connects_aggregated = True
                    b.kind = mv_grid.default_branch_kind_aggregated
                    b.type = mv_grid.default_branch_type_aggregated

                # append to branch list
                edges_graph.append((node1, node2, dict(branch=b)))

                #osmid_branch_dict saves branches overlapping with osmid, format osmid: [branches]
                for osmid in sp:
                    if osmid in osmid_branch_dict: #set due to better performance while iterating later
                        osmid_branch_dict[osmid].add(b)
                    else:
                        osmid_branch_dict[osmid] = {b}

            # add branches to graph
            mv_grid.graph.add_edges_from(edges_graph)

    except:
        logger.exception('unexpected error while converting routing solution to DING0 graph (NetworkX).')

    return mv_grid, osmid_branch_dict


def solve(mv_grid, debug=False, anim=None):

    for load_area in mv_grid.grid_district._lv_load_areas:

        if load_area.is_aggregated:

            ##### initialize parameters for urban mv grid #####

            # import required objects
            mv_station = mv_grid._station
            la, la_centre = load_area, load_area.lv_load_area_centre
            supply_nodes = [mv_load for mv_load in la._mv_loads] + \
                           [lvgd.lv_grid._station for lvgd in la._lv_grid_districts]

            # nodes required to be supplied by urban mv grid
            # get demand (apparent power) and position of supply nodes as dict {ding0_name: value}
            nodes_pos = {node: (node.geo_data.x, node.geo_data.y) for node in supply_nodes}
            nodes_demands = {node: int(node.peak_load /
                                       cfg_ding0.get('assumptions', 'cos_phi_load')) for node in supply_nodes}

            ##### prepare osm graph for routing and stub connections

            # relabel street_graph
            # 1. convert all osmids to str, 2. relabel supply_nodes' osmid with str(ding0_name)
            street_graph, ding0_nodes_map = relabel_graph_nodes(la, cable_dists=None)

            # add mv_station to street_graph
            # reduce street_graph to the least necessary size (supply_nodes, mv_station are kept in graph)
            street_graph = conn_ding0_obj_to_osm_graph(street_graph, mv_station)
            street_graph = reduce_graph_for_dist_matrix_calc(street_graph,
                                                             list(ding0_nodes_map.values()) + [str(mv_station)])

            # split street_graph using k-core
            core_graph, stub_graph = split_graph_by_core(street_graph, mv_station)

            # 11. initial core and stub graph will be adapted based on stub criterion
            # all lv stations / mv loads from stub graph surpassing a load of 1 MVA
            # will be assigned to core graph
            root_nodes = set(stub_graph.nodes) & set(core_graph.nodes)
            node_list = {str(n):n for n in mv_grid.graph.nodes()}
            # 11.1 create dictionary for every stub component of stub graph, 
            # containing all involved nodes: {'comp': {all nodes}, 'root': node,
            # 'load': {node: load}, 'dist': node}
            stub_dict = create_stub_dict(stub_graph, root_nodes, node_list)
            # 11.2 identify nodes to switch by stub criterion
            mod_stubs_list, nodes_to_switch = check_stub_criterion(stub_dict, stub_graph)
            # 11.3 update the stub dict such that just stubs exists fulfilling
            # stub criterion
            stub_dict = update_stub_dict(stub_dict, mod_stubs_list, node_list)
            # 11.4 update graphs respectively
            core_graph, stub_graph = update_graphs(core_graph, stub_graph, nodes_to_switch)

            ##### urban mv routing

            # 12. mv routing is done just for the core graph and related nodes
            # stubs connection will be done later

            # 12.1 prepare routing specs
            specs = osm_graph_to_routing_specs_urban(la, core_graph, mv_station, nodes_pos, nodes_demands)
            # 12.2 create routing graph and solver objects
            RoutingGraph = Graph(specs)
            savings_solver = savings.ClarkeWrightSolver()
            # 12.3 solve problem
            timeout = 300000
            solution = savings_solver.solve(RoutingGraph, timeout)
            # 12.4 transfer solution to ding0 graph
            mv_grid, osmid_branch_dict = routing_solution_to_ding0_graph(mv_grid, core_graph, solution)
            ##### urban mv connect
            mv_grid = mv_urban_connect(mv_grid, street_graph, core_graph, stub_graph, stub_dict, osmid_branch_dict)

    return mv_grid.graph