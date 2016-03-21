import os
import sys

# workaround: add dingo to sys.path to allow imports
PACKAGE_PARENT = '../../..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import time
import matplotlib.pyplot as plt

from dingo.grid.mv_routing.models.models import Graph
from dingo.grid.mv_routing.util import util, data_input
from dingo.grid.mv_routing.solvers import savings, local_search
from dingo.grid.mv_routing.util.distance import calc_geo_distance_vincenty
from dingo.core.network.stations import *


def solve(graph, debug):
    """ Do MV routing for given nodes in `graph`. Translate data from node objects to appropriate format before.

    Args:
        graph:
        debug:

    Returns:

    """

    # build data dictionary from graph nodes for routing
    specs = {}
    nodes_demands = {}
    nodes_pos = {}
    for node in graph.nodes():
        if isinstance(node, StationDingo):
            nodes_pos[str(node)] = (node.geo_data.x, node.geo_data.y)

            if isinstance(node, LVStationDingo):
                nodes_demands[str(node)] = node.grid.region.peak_load_sum
            elif isinstance(node, MVStationDingo):
                nodes_demands[str(node)] = 0
                specs['DEPOT'] = str(node)

    specs['NODE_COORD_SECTION'] = nodes_pos
    specs['DEMAND'] = nodes_demands
    specs['CAPACITY'] = 1000  # TEMP

    specs['MATRIX'] = calc_geo_distance_vincenty(nodes_pos)



    #node_demands =
    #specs['DEMAND'] =
    RoutingGraph = Graph(specs)

    timeout = 30000

    savings_solver = savings.ClarkeWrightSolver()
    local_search_solver = local_search.LocalSearchSolver()

    start = time.time()

    savings_solution = savings_solver.solve(RoutingGraph, timeout)
    elapsed = time.time() - start

    #if not savings_solution.is_complete():
    #    print('=== Solution is not a complete solution! ===')

    print('ClarkeWrightSolver solution:')
    util.print_solution(savings_solution)
    #print('Elapsed time (seconds): {}'.format(elapsed))

    savings_solution.draw_network()

    local_search_solution = local_search_solver.solve(graph, savings_solution, timeout)
    print('Local Search solution:')
    util.print_solution(local_search_solution)
    local_search_solution.draw_network()
