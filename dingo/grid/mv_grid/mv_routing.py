import time

from dingo.grid.mv_grid.models.models import Graph
from dingo.grid.mv_grid.util import util, data_input
from dingo.grid.mv_grid.solvers import savings, local_search
from dingo.grid.mv_grid.util.distance import calc_geo_distance_vincenty
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo
from dingo.tools import config as cfg_dingo

from geopy.distance import vincenty


def dingo_graph_to_routing_specs(graph):
    """ Build data dictionary from graph nodes for routing (translation)

    Args:
        graph: NetworkX graph object with nodes

    Returns:
        specs: Data dictionary for routing, See class `Graph()` in routing's model definition for keys
    """

    specs = {}
    nodes_demands = {}
    nodes_pos = {}
    for node in graph.nodes():

        # station is LV station
        if isinstance(node, LVStationDingo):
            # only major stations are connected via MV ring
            if not node.grid.region.is_satellite:
                nodes_demands[str(node)] = node.grid.region.peak_load_sum
                nodes_pos[str(node)] = (node.geo_data.x, node.geo_data.y)

        # station is MV station
        elif isinstance(node, MVStationDingo):
            nodes_demands[str(node)] = 0
            nodes_pos[str(node)] = (node.geo_data.x, node.geo_data.y)
            specs['DEPOT'] = str(node)

    specs['NODE_COORD_SECTION'] = nodes_pos
    specs['DEMAND'] = nodes_demands
    specs['MATRIX'] = calc_geo_distance_vincenty(nodes_pos)

    # TODO: capacity per MV ring (TEMP) -> Later tech. constraints are used for limitation of ring length
    specs['CAPACITY'] = 3000    # in kW

    return specs


def routing_solution_to_dingo_graph(graph, solution):
    """ Insert `solution` from routing into `graph`

    Args:
        graph: NetworkX graph object with nodes
        solution: Instance of `BaseSolution` or child class (e.g. `LocalSearchSolution`) (=solution from routing)

    Returns:
        graph: NetworkX graph object with nodes and edges
    """
    # TODO: Bisherige Herangehensweise (diese Funktion): Branches werden nach Routing erstellt um die Funktionsfähigkeit
    # TODO: des Routing-Tools auch für die TestCases zu erhalten. Es wird ggf. notwendig, diese direkt im Routing vorzunehmen.

    branch_detour_factor = cfg_dingo.get('assumptions', 'branch_detour_factor')

    # build node dict (name: obj) from graph nodes to map node names on node objects
    node_list = {str(n): n for n in graph.nodes()}

    # add edges from solution to graph
    try:
        depot = solution._nodes[solution._problem._depot.name()]
        for r in solution.routes():
            # build edge list
            n1 = r._nodes[0:len(r._nodes)-1]
            n2 = r._nodes[1:len(r._nodes)]
            edges = list(zip(n1, n2))
            edges.append((depot, r._nodes[0]))
            edges.append((r._nodes[-1], depot))

            # create MV Branch object for every edge in `edges`
            mv_branches = [BranchDingo() for _ in edges]
            edges_with_branches = list(zip(edges, mv_branches))

            # translate solution's node names to graph node objects using dict created before
            # note: branch object is assigned to edge using an attribute ('branch' is used here), it can be accessed
            # using the method `graph_edges()` of class `GridDingo`
            edges_graph = []
            for ((n1, n2), b) in edges_with_branches:
                # get node objects
                node1 = node_list[n1.name()]
                node2 = node_list[n2.name()]
                # set branch length
                b.length = branch_detour_factor *\
                           vincenty((node1.geo_data.x, node1.geo_data.y), (node2.geo_data.x, node2.geo_data.y)).m
                # append to branch list
                edges_graph.append((node1, node2, dict(branch=b)))

            # add branches to graph
            graph.add_edges_from(edges_graph)

    except:
        print('unexpected error while converting routing solution to DINGO graph (NetworkX).')

    return graph

def solve(graph, debug=False, anim=None):
    """ Do MV routing for given nodes in `graph`. Translate data from node objects to appropriate format before.

    Args:
        graph: NetworkX graph object with nodes
        debug: If True, information is printed while routing
        anim: AnimationDingo object (refer to class 'AnimationDingo()' for a more detailed description)

    Returns:
        graph: NetworkX graph object with nodes and edges
    """

    # TODO: Implement debug mode (pass to solver) to get more information while routing (print routes, draw network, ..)

    # translate DINGO graph to routing specs
    specs = dingo_graph_to_routing_specs(graph)

    # create routing graph using specs
    RoutingGraph = Graph(specs)

    timeout = 30000

    # create solver objects
    savings_solver = savings.ClarkeWrightSolver()
    local_search_solver = local_search.LocalSearchSolver()

    start = time.time()

    # create initial solution using Clarke and Wright Savings methods
    savings_solution = savings_solver.solve(RoutingGraph, timeout, debug, anim)

    # OLD, MAY BE USED LATER - Guido, please don't declare a variable later=now() :) :
    #if not savings_solution.is_complete():
    #    print('=== Solution is not a complete solution! ===')

    if debug:
        print('ClarkeWrightSolver solution:')
        util.print_solution(savings_solution)
        print('Elapsed time (seconds): {}'.format(time.time() - start))
        #savings_solution.draw_network()

    # improve initial solution using local search
    local_search_solution = local_search_solver.solve(RoutingGraph, savings_solution, timeout, debug, anim)

    if debug:
        print('Local Search solution:')
        util.print_solution(local_search_solution)
        print('Elapsed time (seconds): {}'.format(time.time() - start))
        local_search_solution.draw_network()

    return routing_solution_to_dingo_graph(graph, local_search_solution)
