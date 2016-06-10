import time

from dingo.grid.mv_grid.models.models import Graph
from dingo.grid.mv_grid.util import util, data_input
from dingo.grid.mv_grid.solvers import savings, local_search
from dingo.grid.mv_grid.util.distance import calc_geo_distance_vincenty
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo, CableDistributorDingo

from shapely.geometry import LineString, Point
from shapely.ops import transform
import pyproj
from functools import partial


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
            edges_graph = [(node_list[n1.name()], node_list[n2.name()], dict(branch=b))
                           for ((n1, n2), b) in edges_with_branches]
            graph.add_edges_from(edges_graph)

    except:
        print('unexpected error while converting routing solution to DINGO graph (NetworkX).')

    return graph

def solve(graph, debug=False):
    """ Do MV routing for given nodes in `graph`. Translate data from node objects to appropriate format before.

    Args:
        graph: NetworkX graph object with nodes
        debug: If True, information is printed while routing

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
    savings_solution = savings_solver.solve(RoutingGraph, timeout)

    # OLD, MAY BE USED LATER - Guido, please don't declare a variable later=now() :) :
    #if not savings_solution.is_complete():
    #    print('=== Solution is not a complete solution! ===')

    if debug:
        print('ClarkeWrightSolver solution:')
        util.print_solution(savings_solution)
        print('Elapsed time (seconds): {}'.format(time.time() - start))
        #savings_solution.draw_network()

    # improve initial solution using local search
    local_search_solution = local_search_solver.solve(RoutingGraph, savings_solution, timeout)

    if debug:
        print('Local Search solution:')
        util.print_solution(local_search_solution)
        print('Elapsed time (seconds): {}'.format(time.time() - start))
        local_search_solution.draw_network()

    return routing_solution_to_dingo_graph(graph, local_search_solution)

def solve_satellites(graph, debug=False):
    """ Connects load areas of type `satellite` (that are not incorporated in cvrp mv routing)

    method:
        1. find nearest line for every satellite using shapely distance
        2.

    Args:
        graph: NetworkX graph object with nodes
        debug: If True, information is printed while routing

    Returns:

    """
    # TODO: change method's name and put to adequate location
    # TODO: method is gonna be used in connection of DES too!

    # WGS84 (conformal) to ETRS (equidistant) projection
    proj1 = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:4326'),  # source coordinate system
            pyproj.Proj(init='epsg:3035'))  # destination coordinate system

    # ETRS (equidistant) to WGS84 (conformal) projection
    proj2 = partial(
            pyproj.transform,
            pyproj.Proj(init='epsg:3035'),  # source coordinate system
            pyproj.Proj(init='epsg:4326'))  # destination coordinate system


    for node in graph.nodes():
        # station is LV station
        if isinstance(node, LVStationDingo):
            # filter major load areas
            #if not node.grid.region.is_satellite:

            # filter satellites
            if node.grid.region.is_satellite:
                #print('node', node)

                satellite_shp = Point(node.geo_data.x, node.geo_data.y)
                satellite_shp = transform(proj1, satellite_shp)
                dist_min = 10**6  # initial distance value

                # calc distance between node and grid's lines -> find nearest line
                for branch in node.grid.region.mv_region.mv_grid.graph_edges():
                    line = branch['adj_nodes']
                    line_shp = LineString([(line[0].geo_data.x, line[0].geo_data.y),
                                           (line[1].geo_data.x, line[1].geo_data.y)])
                    line_shp = transform(proj1, line_shp)

                    dist = satellite_shp.distance(line_shp)
                    if dist < dist_min:
                        dist_min = dist
                        branch_dist_min = branch
                        line_shp_dist_min = line_shp

                # find nearest point on nearest line
                conn_point_shp = line_shp_dist_min.interpolate(line_shp_dist_min.project(satellite_shp))
                conn_point_shp = transform(proj2, conn_point_shp)

                # create cable distributor and add it to grid
                cable_dist = CableDistributorDingo(geo_data=conn_point_shp)
                node.grid.region.mv_region.mv_grid.add_cable_distributor(cable_dist)

                # split old branch into 2 segments (delete old branch and create 2 new ones along cable_dist)
                graph.remove_edge(branch_dist_min['adj_nodes'][0], branch_dist_min['adj_nodes'][1])
                graph.add_edge(branch_dist_min['adj_nodes'][0], cable_dist, branch=BranchDingo())
                graph.add_edge(branch_dist_min['adj_nodes'][1], cable_dist, branch=BranchDingo())

                # add new branch for satellite
                graph.add_edge(node, cable_dist, branch=BranchDingo())

                # TODO: Parametrize new lines!

    return graph