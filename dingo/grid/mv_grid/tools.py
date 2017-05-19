"""This file is part of DINGO, the DIstribution Network GeneratOr.
DINGO is a tool to generate synthetic medium and low voltage power
distribution grids based on open data.

It is developed in the project open_eGo: https://openegoproject.wordpress.com

DINGO lives at github: https://github.com/openego/dingo/
The documentation is available on RTD: http://dingo.readthedocs.io"""

__copyright__  = "Reiner Lemoine Institut gGmbH"
__license__    = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__url__        = "https://github.com/openego/dingo/blob/master/LICENSE"
__author__     = "nesnoj, gplssm"


from dingo.core.network.stations import LVStationDingo
from dingo.core.network import CableDistributorDingo, GeneratorDingo
from dingo.tools.geo import calc_geo_centre_point
import logging


logger = logging.getLogger('dingo')


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
    circuit breaker which is open at normal operation [1]_, [2]_.
    Assuming a ring (route which is connected to the root node at either sides), the optimal position of a circuit
    breaker is defined as the position (virtual cable) between two nodes where the conveyed current is minimal on
    the route. Instead of the peak current, the peak load is used here (assuming a constant voltage).
    
    If a ring is dominated by loads (peak load > peak capacity of generators), only loads are used for determining
    the location of circuit breaker. If generators are prevailing (peak load < peak capacity of generators),
    only generator capacities are considered for relocation.

    The core of this function (calculation of the optimal circuit breaker position) is the same as in
    dingo.grid.mv_grid.models.Route.calc_circuit_breaker_position but here it is
    1. applied to a different data type (NetworkX Graph) and it
    2. adds circuit breakers to all rings.

    The re-location of circuit breakers is necessary because the original position (calculated during routing with
    method mentioned above) shifts during the connection of satellites and therefore it is no longer valid.

    References
    ----------
    .. [1] X. Tao, "Automatisierte Grundsatzplanung von Mittelspannungsnetzen", Dissertation, 2006
    .. [2] FGH e.V.: "Technischer Bericht 302: Ein Werkzeug zur Optimierung der Störungsbeseitigung
        für Planung und Betrieb von Mittelspannungsnetzen", Tech. rep., 2008

    """

    # iterate over all rings and circuit breakers
    for ring, circ_breaker in zip(mv_grid.rings_nodes(include_root_node=False), mv_grid.circuit_breakers()):

        nodes_peak_load = []
        nodes_peak_generation = []

        # iterate over all nodes of ring
        for node in ring:

            # node is LV station -> get peak load and peak generation
            if isinstance(node, LVStationDingo):
                nodes_peak_load.append(node.peak_load)
                nodes_peak_generation.append(node.peak_generation)

            # node is cable distributor -> get all connected nodes of subtree using graph_nodes_from_subtree()
            elif isinstance(node, CableDistributorDingo):
                nodes_subtree = mv_grid.graph_nodes_from_subtree(node)
                nodes_subtree_peak_load = 0
                nodes_subtree_peak_generation = 0

                for node_subtree in nodes_subtree:

                    # node is LV station -> get peak load and peak generation
                    if isinstance(node_subtree, LVStationDingo):
                        nodes_subtree_peak_load += node_subtree.peak_load
                        nodes_subtree_peak_generation += node_subtree.peak_generation

                    # node is LV station -> get peak load and peak generation
                    if isinstance(node_subtree, GeneratorDingo):
                        nodes_subtree_peak_generation += node_subtree.capacity

                nodes_peak_load.append(nodes_subtree_peak_load)
                nodes_peak_generation.append(nodes_subtree_peak_generation)

            else:
                raise ValueError('Ring node has got invalid type.')

        # is ring dominated by load or generation?
        # (check if there's more load than generation in ring or vice versa)
        if sum(nodes_peak_load) > sum(nodes_peak_generation):
            node_peak_data = nodes_peak_load
        else:
            node_peak_data = nodes_peak_generation

        # calc optimal circuit breaker position

        # set init value
        diff_min = 10e6

        # check where difference of demand/generation in two half-rings is minimal
        for ctr in range(len(node_peak_data)):
            # split route and calc demand difference
            route_data_part1 = sum(node_peak_data[0:ctr])
            route_data_part2 = sum(node_peak_data[ctr:len(node_peak_data)])
            diff = abs(route_data_part1 - route_data_part2)

            # equality has to be respected, otherwise comparison stops when demand/generation=0
            if diff <= diff_min:
                diff_min = diff
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
            logger.debug('Ring: {}'.format(ring))
            logger.debug('Circuit breaker {0} was relocated to edge {1}-{2} '
                  '(position on route={3})'.format(
                    circ_breaker, node1, node2, position)
                )
            logger.debug('Peak load sum: {}'.format(sum(nodes_peak_load)))
            logger.debug('Peak loads: {}'.format(nodes_peak_load))
