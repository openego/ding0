from dingo.core.network.stations import LVStationDingo
from dingo.core.network import CableDistributorDingo
from dingo.tools.geo import calc_geo_centre_point


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
    the route. Instead of the peak current, the peak load is used here (assuming a constant voltage).

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

    # iterate over all rings and circuit breakers
    for ring, circ_breaker in zip(mv_grid.rings(include_root_node=False), mv_grid.circuit_breakers()):

        nodes_peak_load = []
        # iterate over all nodes of ring
        for node in ring:

            # node is LV station -> get peak load
            if isinstance(node, LVStationDingo):
                nodes_peak_load.append(node.peak_load)

            # node is cable distributor -> get all connected nodes of subtree using graph_nodes_from_subtree()
            elif isinstance(node, CableDistributorDingo):
                nodes_subtree = mv_grid.graph_nodes_from_subtree(node)
                nodes_subtree_peak_load = 0
                for node_subtree in nodes_subtree:

                    # node is LV station -> get peak load
                    if isinstance(node_subtree, LVStationDingo):
                        nodes_subtree_peak_load += node_subtree.peak_load

                nodes_peak_load.append(nodes_subtree_peak_load)

            else:
                raise ValueError('Ring node has got invalid type.')

        # calc optimal circuit breaker position

        # set init value
        demand_diff_min = 10e6

        # check possible positions in route
        for ctr in range(len(nodes_peak_load)):
            # split route and calc demand difference
            route_demand_part1 = sum(nodes_peak_load[0:ctr])
            route_demand_part2 = sum(nodes_peak_load[ctr:len(nodes_peak_load)])
            demand_diff = abs(route_demand_part1 - route_demand_part2)

            if demand_diff <= demand_diff_min:  # equality has to be respected, otherwise comparison stops when demand=0
                demand_diff_min = demand_diff
                position = ctr
            else:
                break

        # relocate circuit breaker
        node1 = ring[position-1]
        node2 = ring[position]
        circ_breaker.branch = mv_grid._graph.edge[node1][node2]['branch']
        circ_breaker.branch_nodes = (node1, node2)
        circ_breaker.branch.circuit_breaker = circ_breaker
        circ_breaker.geo_data = calc_geo_centre_point(node1, node2)

        if debug:
            print('Ring:', ring)
            print('Circuit breaker', circ_breaker, 'was relocated to edge', node1, '-', node2,
                  '(position on route=', position, ')')
            print('Peak load sum:', sum(nodes_peak_load))
            print('Peak loads:', nodes_peak_load)
