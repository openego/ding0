

def set_circuit_breakers(mv_grid, debug=False):
    """ Calculates the optimal position of a circuit breaker on all routes of mv_grid, adds and connects them to graph.
    Args:
        mv_grid: MVGridDingo object
        debug: If True, information is printed during process
    Returns:
        nothing

    Notes
    -----
    According to planning principles of MV grids, a MV ring is run as two strings (half-rings) separated by a
    circuit breaker which is open at normal operation.
    Assuming a ring (route which is connected to the root node at either sides), the optimal position of a circuit
    breaker is defined as the position (virtual cable) between two nodes where the conveyed current is minimal on
    the route.

    The core of this function (calculation of the optimal circuit breaker position) is the same as in
    dingo.grid.mv_grid.models.Route.calc_circuit_breaker_position but here it is
    1. applied to a different data type (NetworkX Graph) and it
    2. adds circuit breakers to all rings.

    The re-location of circuit breakers is necessary because the original position (calculated during routing with
    method mentioned above) shifts during the connection of satellites and therefore it is no longer valid.

    References
    ----------

    """
    # TODO: add references (Tao)

    # set init value
    demand_diff_min = 10e6

    # # remove old circuit breakers if existent
    # if mv_grid.circuit_breakers_count > 0:
    #     for circ_breaker in mv_grid.circuit_breakers():
    #         # add new branch (bypass circuit breaker)
    #         nodes = mv_grid.graph_nodes_from_branch(circ_breaker.branch)
    #         # remove circuit breaker
    #         mv_grid._graph.remove_node(circ_breaker)

    for ring, circ_breaker in zip(mv_grid.rings(), mv_grid.circuit_breakers()):
        print(ring, circ_breaker)
        for node in ring:
            if isinstance(node, LVStationDingo):


    # # check possible positions in route
    # for ctr in range(len(self._nodes)):
    #     # split route and calc demand difference
    #     route_demand_part1 = sum([node.demand() for node in self._nodes[0:ctr]])
    #     route_demand_part2 = sum([node.demand() for node in self._nodes[ctr:len(self._nodes)]])
    #     demand_diff = abs(route_demand_part1 - route_demand_part2)
    #
    #     if demand_diff < demand_diff_min:
    #         demand_diff_min = demand_diff
    #         position = ctr
    #
    # if debug:
    #     print('sum 1=', sum([node.demand() for node in self._nodes[0:position]]))
    #     print('sum 2=', sum([node.demand() for node in self._nodes[position:len(self._nodes)]]))
    #     print('Position of circuit breaker: ', self._nodes[position-1], '-', self._nodes[position],
    #           '(sumdiff=', demand_diff_min, ')')
    #
    # return position