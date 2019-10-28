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


from ding0.core.network.stations import LVStationDing0
from ding0.core.network import CableDistributorDing0, GeneratorDing0, LoadDing0
from ding0.tools.geo import calc_geo_centre_point
from ding0.tools import config as cfg_ding0
import logging


logger = logging.getLogger('ding0')


def set_circuit_breakers(mv_grid, mode='load', debug=False):
    """ Calculates the optimal position of a circuit breaker at lv stations (if existing)
    on all routes of mv_grid, adds and connects them to graph.
    
    Args
    ----
    mv_grid: MVGridDing0
       MV grid instance
    debug: bool, defaults to False
       If True, information is printed during process
    

    Note
    -----
    According to planning principles of MV grids, a MV ring is run as two strings (half-rings) separated by a
    circuit breaker which is open at normal operation [#]_, [#]_.
    Assuming a ring (route which is connected to the root node at either sides), the optimal position of a circuit
    breaker is defined as the position (virtual cable) between two nodes where the conveyed current is minimal on
    the route. Instead of the peak current, the peak load is used here (assuming a constant voltage).
    
    The circuit breaker will be installed to a LV station, unless none
    exists in a ring. In this case, a node of arbitrary type is chosen for the
    location of the switch disconnecter.
    
    If a ring is dominated by loads (peak load > peak capacity of generators), only loads are used for determining
    the location of circuit breaker. If generators are prevailing (peak load < peak capacity of generators),
    only generator capacities are considered for relocation.

    The core of this function (calculation of the optimal circuit breaker position) is the same as in
    ding0.grid.mv_grid.models.Route.calc_circuit_breaker_position but here it is
    1. applied to a different data type (NetworkX Graph) and it
    2. adds circuit breakers to all rings.

    The re-location of circuit breakers is necessary because the original position (calculated during routing with
    method mentioned above) shifts during the connection of satellites and therefore it is no longer valid.

    References
    ----------
    .. [#] X. Tao, "Automatisierte Grundsatzplanung von Mittelspannungsnetzen", Dissertation, 2006
    .. [#] FGH e.V.: "Technischer Bericht 302: Ein Werkzeug zur Optimierung der Störungsbeseitigung
        für Planung und Betrieb von Mittelspannungsnetzen", Tech. rep., 2008

    """

    def relocate_circuit_breaker():
        """
        Moves circuit breaker to different position in ring.

            
        Note
        -----
        Branch of circuit breaker should be set to None in advance. 
        So far only useful to relocate all circuit breakers in a grid as the 
        position of the inserted circuit breaker is not checked beforehand. If
        used for single circuit breakers make sure to insert matching ring and
        circuit breaker.
        """
        node_cb = ring[position]
        # check if node is last node of ring
        if position < len(ring):
            # check which branch to disconnect by determining load difference
            # of neighboring nodes
            diff2 = abs(sum(node_peak_data[0:position+1]) -
                        sum(node_peak_data[position+1:len(node_peak_data)]))
            if diff2 < diff_min:
                node2 = ring[position+1]
            else:
                node2 = ring[position-1]
        else:
            node2 = ring[position-1]

        circ_breaker.branch = mv_grid.graph.adj[node_cb][node2]['branch']
        circ_breaker.branch_nodes = (node_cb, node2)
        circ_breaker.switch_node = node_cb
        circ_breaker.branch.circuit_breaker = circ_breaker
        circ_breaker.geo_data = calc_geo_centre_point(node_cb, node2)


    # get power factor for loads and generators
    cos_phi_load = cfg_ding0.get('assumptions', 'cos_phi_load')
    cos_phi_feedin = cfg_ding0.get('assumptions', 'cos_phi_gen')
    
    # "disconnect" circuit breakers from branches
    for cb in mv_grid.circuit_breakers():
        cb.branch.circuit_breaker = None

    # iterate over all rings and circuit breakers
    for ring, circ_breaker in zip(mv_grid.rings_nodes(include_root_node=False), 
                                  mv_grid.circuit_breakers()):

        nodes_peak_load = []
        nodes_peak_generation = []

        # iterate over all nodes of ring
        for node in ring:

            # node is LV station -> get peak load and peak generation
            if isinstance(node, LVStationDing0):
                nodes_peak_load.append(node.peak_load / cos_phi_load)
                nodes_peak_generation.append(
                    node.peak_generation / cos_phi_feedin)

            # node is cable distributor -> get all connected nodes of subtree using graph_nodes_from_subtree()
            elif isinstance(node, CableDistributorDing0):
                nodes_subtree = mv_grid.graph_nodes_from_subtree(node)
                nodes_subtree_peak_load = 0
                nodes_subtree_peak_generation = 0

                for node_subtree in nodes_subtree:

                    # node is LV station -> get peak load and peak generation
                    if isinstance(node_subtree, LVStationDing0):
                        nodes_subtree_peak_load += node_subtree.peak_load / \
                                                   cos_phi_load
                        nodes_subtree_peak_generation += node_subtree.peak_generation / \
                                                         cos_phi_feedin

                    # node is LV station -> get peak load and peak generation
                    if isinstance(node_subtree, GeneratorDing0):
                        nodes_subtree_peak_generation += node_subtree.capacity / \
                                                         cos_phi_feedin

                nodes_peak_load.append(nodes_subtree_peak_load)
                nodes_peak_generation.append(nodes_subtree_peak_generation)

            else:
                raise ValueError('Ring node has got invalid type.')

        if mode == 'load':
            node_peak_data = nodes_peak_load
        elif mode == 'generation':
            node_peak_data = nodes_peak_generation
        elif mode == 'loadgen':
            # is ring dominated by load or generation?
            # (check if there's more load than generation in ring or vice versa)
            if sum(nodes_peak_load) > sum(nodes_peak_generation):
                node_peak_data = nodes_peak_load
            else:
                node_peak_data = nodes_peak_generation
        else:
            raise ValueError('parameter \'mode\' is invalid!')

        # calc optimal circuit breaker position
        # Set start value for difference in ring halfs
        diff_min = 10e9

        # if none of the nodes is of the type LVStation, a switch
        # disconnecter will be installed anyways.
        if any([isinstance(n, LVStationDing0) for n in ring]):
            has_lv_station = True
        else:
            has_lv_station = False
            logging.debug("Ring {} does not have a LV station. "
                          "Switch disconnecter is installed at arbitrary "
                          "node.".format(ring))

        # check where difference of demand/generation in two half-rings is minimal
        for ctr in range(len(node_peak_data)):
            # check if node that owns the switch disconnector is of type
            # LVStation
            if isinstance(ring[ctr], LVStationDing0) or not has_lv_station:
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
        relocate_circuit_breaker()

        if debug:
            logger.debug('Ring: {}'.format(ring))
            logger.debug('Circuit breaker {0} was relocated to edge {1} '
                  '(position on route={2})'.format(
                    circ_breaker, repr(circ_breaker.branch), position)
                )
            logger.debug('Peak load sum: {}'.format(sum(nodes_peak_load)))
            logger.debug('Peak loads: {}'.format(nodes_peak_load))



