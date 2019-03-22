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


import os
import pyproj
from functools import partial
import time
import logging

from ding0.core.network.stations import *
from ding0.core.network import BranchDing0, GeneratorDing0
from ding0.core import MVCableDistributorDing0
from ding0.core.structure.groups import LoadAreaGroupDing0
from ding0.core.structure.regions import LVLoadAreaCentreDing0
from ding0.tools import config as cfg_ding0
from ding0.tools.geo import calc_geo_branches_in_buffer,calc_geo_dist_vincenty,\
                            calc_geo_centre_point, calc_geo_branches_in_polygon

if not 'READTHEDOCS' in os.environ:
    from shapely.geometry import LineString
    from shapely.ops import transform

logger = logging.getLogger('ding0')


def find_nearest_conn_objects(node_shp, branches, proj, conn_dist_weight, debug, branches_only=False):
    """Searches all `branches` for the nearest possible connection object per branch.
    
    Picks out 1 object out of 3 possible objects: 
    
    * 2 branch-adjacent stations and
    * 1 potentially created cable distributor on the line (perpendicular projection)).
    
    The resulting stack (list) is sorted ascending by distance from node.

    Parameters
    ----------
    node_shp: :shapely:`Shapely Point object<points>`
        Shapely Point object of node
    branches: BranchDing0
        BranchDing0 objects of MV region
    proj: :pyproj:`pyproj Proj object< >`
        nodes' CRS to equidistant CRS (e.g. WGS84 -> ETRS)
    conn_dist_weight: float
        length weighting to prefer stations instead of direct line connection.
    debug: bool
        If True, information is printed during process
    branches_only: bool, defaults to False
        If True, only branch objects are considered as connection objects

    Returns
    -------
    :obj:`list`
        List of connection objects. 
        Each object is represented by dict with Ding0 object,
        shapely object, and distance to node.
    
    See Also
    --------
    mv_connect_satellites : for details on `conn_dist_weight` param
    """

    # threshold which is used to determine if 2 objects are on the same position (see below for details on usage)
    conn_diff_tolerance = cfg_ding0.get('mv_routing', 'conn_diff_tolerance')

    conn_objects_min_stack = []

    for branch in branches:
        stations = branch['adj_nodes']

        # create shapely objects for 2 stations and line between them, transform to equidistant CRS
        station1_shp = transform(proj, stations[0].geo_data)
        station2_shp = transform(proj, stations[1].geo_data)
        line_shp = LineString([station1_shp, station2_shp])

        # create dict with DING0 objects (line & 2 adjacent stations), shapely objects and distances
        if not branches_only:
            conn_objects = {'s1': {'obj': stations[0],
                                   'shp': station1_shp,
                                   'dist': node_shp.distance(station1_shp) * conn_dist_weight * 0.999},
                            's2': {'obj': stations[1],
                                   'shp': station2_shp,
                                   'dist': node_shp.distance(station2_shp) * conn_dist_weight * 0.999},
                            'b': {'obj': branch,
                                  'shp': line_shp,
                                  'dist': node_shp.distance(line_shp)}}

            # Remove branch from the dict of possible conn. objects if it is too close to a node.
            # Without this solution, the target object is not unique for different runs (and so
            # were the topology)
            if (
                    abs(conn_objects['s1']['dist'] - conn_objects['b']['dist']) < conn_diff_tolerance
                 or abs(conn_objects['s2']['dist'] - conn_objects['b']['dist']) < conn_diff_tolerance
               ):
                del conn_objects['b']

            # remove MV station as possible connection point
            if isinstance(conn_objects['s1']['obj'], MVStationDing0):
                del conn_objects['s1']
            elif isinstance(conn_objects['s2']['obj'], MVStationDing0):
                del conn_objects['s2']

        else:
            conn_objects = {'b': {'obj': branch,
                                  'shp': line_shp,
                                  'dist': node_shp.distance(line_shp)}}



        # find nearest connection point on given triple dict (2 branch-adjacent stations + cable dist. on line)
        conn_objects_min = min(conn_objects.values(), key=lambda v: v['dist'])
        #if not branches_only:
        #    conn_objects_min_stack.append(conn_objects_min)
        #elif isinstance(conn_objects_min['shp'], LineString):
        #    conn_objects_min_stack.append(conn_objects_min)
        conn_objects_min_stack.append(conn_objects_min)

    # sort all objects by distance from node
    conn_objects_min_stack = [_ for _ in sorted(conn_objects_min_stack, key=lambda x: x['dist'])]

    if debug:
        logger.debug('Stack length: {}'.format(len(conn_objects_min_stack)))

    return conn_objects_min_stack


def get_lv_load_area_group_from_node_pair(node1, node2):

    lv_load_area_group = None

    # both nodes are LV stations -> get group from 1 or 2
    if (isinstance(node1, LVLoadAreaCentreDing0) and
       isinstance(node2, LVLoadAreaCentreDing0)):
        if not node1.lv_load_area.lv_load_area_group:
            lv_load_area_group = node2.lv_load_area.lv_load_area_group
        else:
            lv_load_area_group = node1.lv_load_area.lv_load_area_group

    # node 1 is LV station and node 2 not -> get group from node 1
    elif (isinstance(node1, LVLoadAreaCentreDing0) and
          isinstance(node2, (MVStationDing0, MVCableDistributorDing0))):
        lv_load_area_group = node1.lv_load_area.lv_load_area_group

    # node 2 is LV station and node 1 not -> get group from node 2
    elif (isinstance(node1, (MVStationDing0, MVCableDistributorDing0)) and
          isinstance(node2, LVLoadAreaCentreDing0)):
        lv_load_area_group = node2.lv_load_area.lv_load_area_group

    # both nodes are not a LV station -> no group
    elif (isinstance(node1, (MVStationDing0, MVCableDistributorDing0)) and
          isinstance(node2, (MVStationDing0, MVCableDistributorDing0))):
        lv_load_area_group = None

    return lv_load_area_group


def find_connection_point(node, node_shp, graph, proj, conn_objects_min_stack, conn_dist_ring_mod, debug):
    """ Goes through the possible target connection objects in `conn_objects_min_stack` (from nearest to most far
        object) and tries to connect `node` to one of them.

    Function searches from nearest to most far object.
    
    Parameters
    ----------
    node: LVLoadAreaCentreDing0, i.e.
        Origin node - Ding0 graph object (e.g. LVLoadAreaCentreDing0)
    node_shp: :shapely:`Shapely Point object<points>`
        Shapely Point object of node
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes
    proj: :pyproj:`pyproj Proj object< >`
        equidistant CRS to conformal CRS (e.g. ETRS -> WGS84)
    conn_objects_min_stack: :obj:`list`
        List of connection objects. 
        
        Each object is represented by dict with Ding0 object, shapely object,
        and distance to node, sorted ascending by distance.
    conn_dist_ring_mod: type
        Max. distance when nodes are included into route instead of creating a 
        new line.
    debug: bool
        If True, information is printed during process

    See Also
    --------
    ding0.grid.mv_grid.mv_connect : for details on the `conn_dist_ring_mod` parameter.
    """

    node_connected = False

    # go through the stack (from nearest to most far connection target object)
    for dist_min_obj in conn_objects_min_stack:

        nodes_are_members_of_ring = False

        # target object is branch
        if isinstance(dist_min_obj['shp'], LineString):
            # rename for readability
            node1 = dist_min_obj['obj']['adj_nodes'][0]
            node2 = dist_min_obj['obj']['adj_nodes'][1]

            lv_load_area_group = get_lv_load_area_group_from_node_pair(node1, node2)

            # check if target branch belongs to a main ring
            nodes_are_members_of_ring = any(node1 in ring and node2 in ring for ring in node.grid.rings_nodes())
            branch_ring = dist_min_obj['obj']['branch'].ring

        # target object is node
        else:
            if isinstance(dist_min_obj['obj'], MVCableDistributorDing0):
                lv_load_area_group = dist_min_obj['obj'].lv_load_area_group
            else:
                lv_load_area_group = dist_min_obj['obj'].lv_load_area.lv_load_area_group

        # target object doesn't belong to a satellite string (is not a member of a Load Area group)
        if not lv_load_area_group:

            # connect node
            target_obj_result = connect_node(node,
                                             node_shp,
                                             node.lv_load_area.mv_grid_district.mv_grid,
                                             dist_min_obj,
                                             proj,
                                             graph,
                                             conn_dist_ring_mod,
                                             debug)

            # if node was connected via branch (target line not re-routed and not member of aggregated load area):
            # create new LV load_area group for current node
            if (target_obj_result is not None) and (target_obj_result != 're-routed'):
                lv_load_area_group = LoadAreaGroupDing0(mv_grid_district=node.lv_load_area.mv_grid_district,
                                                        root_node=target_obj_result)
                lv_load_area_group.add_lv_load_area(lv_load_area=node.lv_load_area)
                node.lv_load_area.lv_load_area_group = lv_load_area_group
                node.lv_load_area.mv_grid_district.add_lv_load_area_group(lv_load_area_group)

                if debug:
                    logger.debug('New LV load_area group {} created!'.format(
                        lv_load_area_group))

                # node connected, stop connection for current node
                node_connected = True
                break

            # node was inserted into line (target line was re-routed)
            elif target_obj_result == 're-routed':

                # if main ring was re-routed to include node => node is not a satellite anymore
                if nodes_are_members_of_ring:
                    node.lv_load_area.is_satellite = False
                    node.lv_load_area.ring = branch_ring

                # node connected, stop connection for current node
                node_connected = True
                break

        # target object is member of a Load Area group
        else:

            # connect node
            target_obj_result = connect_node(node,
                                             node_shp,
                                             node.lv_load_area.mv_grid_district.mv_grid,
                                             dist_min_obj,
                                             proj,
                                             graph,
                                             conn_dist_ring_mod,
                                             debug)

            # if node was connected via branch (target line not re-routed and not member of aggregated load area):
            # create new LV load_area group for current node
            if (target_obj_result is not None) and (target_obj_result != 're-routed'):
                # node can join LV load_area group
                if lv_load_area_group.can_add_lv_load_area(node=node):

                    # add node to LV load_area group
                    lv_load_area_group.add_lv_load_area(lv_load_area=node.lv_load_area)
                    node.lv_load_area.lv_load_area_group = lv_load_area_group

                    if isinstance(target_obj_result, MVCableDistributorDing0):
                        lv_load_area_group.add_lv_load_area(lv_load_area=target_obj_result)
                        target_obj_result.lv_load_area_group = lv_load_area_group

                    if debug:
                        logger.debug('LV load_area group {} joined!'.format(
                            lv_load_area_group))

                    # node connected, stop connection for current node
                    node_connected = True
                    break

                # cannot join LV load_area group
                else:
                    if debug:
                        logger.debug('Node {0} could not be added to '
                                     'load_area group {1}'.format(
                            node, lv_load_area_group))

                    # rollback changes in graph
                    disconnect_node(node, target_obj_result, graph, debug)

                    # continue with next possible connection point
                    continue

            # node was inserted into line (target line was re-routed)
            elif target_obj_result == 're-routed':
                # add node to LV load_area group
                lv_load_area_group.add_lv_load_area(lv_load_area=node.lv_load_area)
                node.lv_load_area.lv_load_area_group = lv_load_area_group

                # if main ring was re-routed to include node => node is not a satellite anymore
                if nodes_are_members_of_ring:
                    node.lv_load_area.is_satellite = False
                    node.lv_load_area.ring = branch_ring

                # node inserted into existing route, stop connection for current node
                node_connected = True
                break

            # else: node could not be connected because target object belongs to load area of aggregated type

    if not node_connected and debug:
        logger.debug(
            'Node {} could not be connected, try to increase the parameter '
            '`load_area_sat_buffer_radius` in config file `config_calc.cfg` '
            'to gain more possible connection points.'.format(node))


def connect_node(node, node_shp, mv_grid, target_obj, proj, graph, conn_dist_ring_mod, debug):
    """ Connects `node` to `target_obj`.

    Parameters
    ----------
    node: LVLoadAreaCentreDing0, i.e.
        Origin node - Ding0 graph object (e.g. LVLoadAreaCentreDing0)
    node_shp: :shapely:`Shapely Point object<points>`
        Shapely Point object of origin node
    target_obj: type
        object that node shall be connected to
    proj: :pyproj:`pyproj Proj object< >`
        equidistant CRS to conformal CRS (e.g. ETRS -> WGS84)
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes and newly created branches
    conn_dist_ring_mod: float
        Max. distance when nodes are included into route instead of creating a 
        new line.
    debug: bool
        If True, information is printed during process.

    Returns
    -------
    :obj:`LVLoadAreaCentreDing0`
        object that node was connected to.
        
        (instance of :obj:`LVLoadAreaCentreDing0` or :obj:`MVCableDistributorDing0`.
        
        If node is included into line instead of creating a new line (see arg
        `conn_dist_ring_mod`), `target_obj_result` is None.
                           
    See Also
    --------
    ding0.grid.mv_grid.mv_connect : for details on the `conn_dist_ring_mod` parameter.
    """

    target_obj_result = None

    # MV line is nearest connection point
    if isinstance(target_obj['shp'], LineString):

        adj_node1 = target_obj['obj']['adj_nodes'][0]
        adj_node2 = target_obj['obj']['adj_nodes'][1]

        # find nearest point on MV line
        conn_point_shp = target_obj['shp'].interpolate(target_obj['shp'].project(node_shp))
        conn_point_shp = transform(proj, conn_point_shp)

        # target MV line does currently not connect a load area of type aggregated
        if not target_obj['obj']['branch'].connects_aggregated:

            # Node is close to line
            # -> insert node into route (change existing route)
            if (target_obj['dist'] < conn_dist_ring_mod):
                # backup kind and type of branch
                branch_type = graph.edge[adj_node1][adj_node2]['branch'].type
                branch_kind = graph.edge[adj_node1][adj_node2]['branch'].kind
                branch_ring = graph.edge[adj_node1][adj_node2]['branch'].ring

                # check if there's a circuit breaker on current branch,
                # if yes set new position between first node (adj_node1) and newly inserted node
                circ_breaker = graph.edge[adj_node1][adj_node2]['branch'].circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = calc_geo_centre_point(adj_node1, node)

                # split old ring main route into 2 segments (delete old branch and create 2 new ones
                # along node)
                graph.remove_edge(adj_node1, adj_node2)

                branch_length = calc_geo_dist_vincenty(adj_node1, node)
                branch = BranchDing0(length=branch_length,
                                     circuit_breaker=circ_breaker,
                                     kind=branch_kind,
                                     type=branch_type,
                                     ring=branch_ring)
                if circ_breaker is not None:
                    circ_breaker.branch = branch
                graph.add_edge(adj_node1, node, branch=branch)

                branch_length = calc_geo_dist_vincenty(adj_node2, node)
                graph.add_edge(adj_node2, node, branch=BranchDing0(length=branch_length,
                                                                   kind=branch_kind,
                                                                   type=branch_type,
                                                                   ring=branch_ring))

                target_obj_result = 're-routed'

                if debug:
                    logger.debug('Ring main route modified to include '
                                 'node {}'.format(node))

            # Node is too far away from route
            # => keep main route and create new line from node to (cable distributor on) route.
            else:

                # create cable distributor and add it to grid
                cable_dist = MVCableDistributorDing0(geo_data=conn_point_shp,
                                                     grid=mv_grid)
                mv_grid.add_cable_distributor(cable_dist)

                # check if there's a circuit breaker on current branch,
                # if yes set new position between first node (adj_node1) and newly created cable distributor
                circ_breaker = graph.edge[adj_node1][adj_node2]['branch'].circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = calc_geo_centre_point(adj_node1, cable_dist)

                # split old branch into 2 segments (delete old branch and create 2 new ones along cable_dist)
                # ===========================================================================================

                # backup kind and type of branch
                branch_kind = graph.edge[adj_node1][adj_node2]['branch'].kind
                branch_type = graph.edge[adj_node1][adj_node2]['branch'].type
                branch_ring = graph.edge[adj_node1][adj_node2]['branch'].ring

                graph.remove_edge(adj_node1, adj_node2)

                branch_length = calc_geo_dist_vincenty(adj_node1, cable_dist)
                branch = BranchDing0(length=branch_length,
                                     circuit_breaker=circ_breaker,
                                     kind=branch_kind,
                                     type=branch_type,
                                     ring=branch_ring)
                if circ_breaker is not None:
                    circ_breaker.branch = branch
                graph.add_edge(adj_node1, cable_dist, branch=branch)

                branch_length = calc_geo_dist_vincenty(adj_node2, cable_dist)
                graph.add_edge(adj_node2, cable_dist, branch=BranchDing0(length=branch_length,
                                                                         kind=branch_kind,
                                                                         type=branch_type,
                                                                         ring=branch_ring))

                # add new branch for satellite (station to cable distributor)
                # ===========================================================

                # get default branch kind and type from grid to use it for new branch
                branch_kind = mv_grid.default_branch_kind
                branch_type = mv_grid.default_branch_type

                branch_length = calc_geo_dist_vincenty(node, cable_dist)
                graph.add_edge(node, cable_dist, branch=BranchDing0(length=branch_length,
                                                                    kind=branch_kind,
                                                                    type=branch_type,
                                                                    ring=branch_ring))
                target_obj_result = cable_dist

                # debug info
                if debug:
                    logger.debug('Nearest connection point for object {0} '
                                 'is branch {1} (distance={2} m)'.format(
                        node, target_obj['obj']['adj_nodes'], target_obj['dist']))

    # node ist nearest connection point
    else:

        # what kind of node is to be connected? (which type is node of?)
        #   LVLoadAreaCentreDing0: Connect to LVLoadAreaCentreDing0 only
        #   LVStationDing0: Connect to LVLoadAreaCentreDing0, LVStationDing0 or MVCableDistributorDing0
        #   GeneratorDing0: Connect to LVLoadAreaCentreDing0, LVStationDing0, MVCableDistributorDing0 or GeneratorDing0
        if isinstance(node, LVLoadAreaCentreDing0):
            valid_conn_objects = LVLoadAreaCentreDing0
        elif isinstance(node, LVStationDing0):
            valid_conn_objects = (LVLoadAreaCentreDing0, LVStationDing0, MVCableDistributorDing0)
        elif isinstance(node, GeneratorDing0):
            valid_conn_objects = (LVLoadAreaCentreDing0, LVStationDing0, MVCableDistributorDing0, GeneratorDing0)
        else:
            raise ValueError('Oops, the node you are trying to connect is not a valid connection object')

        # if target is Load Area centre or LV station, check if it belongs to a load area of type aggregated
        # (=> connection not allowed)
        if isinstance(target_obj['obj'], (LVLoadAreaCentreDing0, LVStationDing0)):
            target_is_aggregated = target_obj['obj'].lv_load_area.is_aggregated
        else:
            target_is_aggregated = False

        # target node is not a load area of type aggregated
        if isinstance(target_obj['obj'], valid_conn_objects) and not target_is_aggregated:

            # get default branch kind and type from grid to use it for new branch
            branch_kind = mv_grid.default_branch_kind
            branch_type = mv_grid.default_branch_type

            # get branch ring obj
            branch_ring = mv_grid.get_ring_from_node(target_obj['obj'])

            # add new branch for satellite (station to station)
            branch_length = calc_geo_dist_vincenty(node, target_obj['obj'])
            graph.add_edge(node, target_obj['obj'], branch=BranchDing0(length=branch_length,
                                                                       kind=branch_kind,
                                                                       type=branch_type,
                                                                       ring=branch_ring))
            target_obj_result = target_obj['obj']

            # debug info
            if debug:
                logger.debug('Nearest connection point for object {0} is station {1} '
                      '(distance={2} m)'.format(
                    node, target_obj['obj'], target_obj['dist']))

    return target_obj_result


def disconnect_node(node, target_obj_result, graph, debug):
    """ Disconnects `node` from `target_obj`

    Parameters
    ----------
    node: LVLoadAreaCentreDing0, i.e.
        Origin node - Ding0 graph object (e.g. LVLoadAreaCentreDing0)
    target_obj_result: LVLoadAreaCentreDing0, i.e.
        Origin node - Ding0 graph object (e.g. LVLoadAreaCentreDing0)
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes and newly created branches
    debug: bool
        If True, information is printed during process

    """

    # backup kind and type of branch
    branch_kind = graph.edge[node][target_obj_result]['branch'].kind
    branch_type = graph.edge[node][target_obj_result]['branch'].type
    branch_ring = graph.edge[node][target_obj_result]['branch'].ring

    graph.remove_edge(node, target_obj_result)

    if isinstance(target_obj_result, MVCableDistributorDing0):

        neighbor_nodes = graph.neighbors(target_obj_result)

        if len(neighbor_nodes) == 2:
            node.grid.remove_cable_distributor(target_obj_result)

            branch_length = calc_geo_dist_vincenty(neighbor_nodes[0], neighbor_nodes[1])
            graph.add_edge(neighbor_nodes[0], neighbor_nodes[1], branch=BranchDing0(length=branch_length,
                                                                                    kind=branch_kind,
                                                                                    type=branch_type,
                                                                                    ring=branch_ring))

    if debug:
        logger.debug('disconnect edge {0}-{1}'.format(node, target_obj_result))


def parametrize_lines(mv_grid):
    """ Set unparametrized branches to default branch type
    
    Parameters
    ----------
    mv_grid: MVGridDing0
        MV grid instance

    Notes
    -----
    During the connection process of satellites, new branches are created - 
    these have to be parametrized.
    """

    for branch in mv_grid.graph_edges():
        if branch['branch'].kind is None:
            branch['branch'].kind = mv_grid.default_branch_kind
        if branch['branch'].type is None:
            branch['branch'].type = mv_grid.default_branch_type


def mv_connect_satellites(mv_grid, graph, mode='normal', debug=False):
    """ Connect satellites (small Load Areas) to MV grid

    Parameters
    ----------
    mv_grid: MVGridDing0
        MV grid instance
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes
    mode: :obj:`str`, defaults to 'normal'
        Specify mode how satellite `LVLoadAreaCentreDing0` are connected to the
        grid. Mode normal (default) considers for restrictions like max.
        string length, max peak load per string.
        The mode 'isolated' disregards any connection restrictions and connects
        the node `LVLoadAreaCentreDing0` to the next connection point.
        
    debug: bool, defaults to False
         If True, information is printed during process

    Notes
    -----
    conn_dist_weight: The satellites can be connected to line (new terminal is
    created) or to one station where the line ends, depending on the distance
    from satellite to the objects. This threshold is a length weighting to
    prefer stations instead of direct line connection to respect grid planning
    principles.

    Example: The distance from satellite to line is 1km, to station1 1.2km, to
    station2 2km. With conn_dist_threshold=0.75, the 'virtual' distance to
    station1 would be 1.2km * 0.75 = 0.9km, so this conn. point would be
    preferred.

    Returns
    -------
    :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes and newly created branches
    """

    # conn_dist_weight: The satellites can be connected to line (new terminal is created) or to one station where the
    # line ends, depending on the distance from satellite to the objects. This threshold is a length weighting to prefer
    # stations instead of direct line connection to respect grid planning principles.
    # Example: The distance from satellite to line is 1km, to station1 1.2km, to station2 2km.
    # With conn_dist_threshold=0.75, the 'virtual' distance to station1 would be 1.2km * 0.75 = 0.9km, so this conn.
    # point would be preferred.
    conn_dist_weight = cfg_ding0.get('mv_connect', 'load_area_sat_conn_dist_weight')

    # conn_dist_ring_mod: Allow re-routing of ring main route if node is closer than this threshold (in m) to ring.
    conn_dist_ring_mod = cfg_ding0.get('mv_connect', 'load_area_sat_conn_dist_ring_mod')

    load_area_sat_buffer_radius = cfg_ding0.get('mv_connect', 'load_area_sat_buffer_radius')
    load_area_sat_buffer_radius_inc = cfg_ding0.get('mv_connect', 'load_area_sat_buffer_radius_inc')

    start = time.time()

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

    # check all nodes
    if mode == 'normal':
        #nodes = sorted(graph.nodes(), key=lambda x: repr(x))
        nodes = mv_grid.graph_isolated_nodes()
    elif mode == 'isolated':
        nodes = mv_grid.graph_isolated_nodes()
    else:
        raise ValueError('\'mode\' is invalid.')

    for node in nodes:

        # node is Load Area centre
        if isinstance(node, LVLoadAreaCentreDing0):

            # satellites only
            if node.lv_load_area.is_satellite:

                node_shp = transform(proj1, node.geo_data)

                if mode == 'normal':
                    # get branches within a the predefined radius `load_area_sat_buffer_radius`
                    branches = calc_geo_branches_in_buffer(node,
                                                           mv_grid,
                                                           load_area_sat_buffer_radius,
                                                           load_area_sat_buffer_radius_inc, proj1)
                elif mode == 'isolated':
                    # get nodes of all MV rings
                    nodes = set()
                    [nodes.update(ring_nodes) for ring_nodes in list(mv_grid.rings_nodes(include_root_node=True))]
                    nodes = list(nodes)
                    # get branches of these nodes
                    branches = []
                    [branches.append(mv_grid.graph_branches_from_node(node_branches)) for node_branches in nodes]
                    # reformat branches
                    branches = [_ for _ in list(mv_grid.graph_edges())
                                if (_['adj_nodes'][0] in nodes and _['adj_nodes'][1] in nodes)]

                # calc distance between node and grid's lines -> find nearest line
                conn_objects_min_stack = find_nearest_conn_objects(node_shp, branches, proj1,
                                                                   conn_dist_weight, debug,
                                                                   branches_only=False)

                # iterate over object stack
                find_connection_point(node, node_shp, graph, proj2, conn_objects_min_stack,
                                      conn_dist_ring_mod, debug)

    # parametrize newly created branches
    parametrize_lines(mv_grid)

    if debug:
        logger.debug('Elapsed time (mv_connect): {}'.format(time.time() - start))

    return graph


def mv_connect_stations(mv_grid_district, graph, debug=False):
    """ Connect LV stations to MV grid

    Parameters
    ----------
    mv_grid_district: MVGridDistrictDing0
        MVGridDistrictDing0 object for which the connection process has to be done
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes
    debug: bool, defaults to False
        If True, information is printed during process

    Returns
    -------
    :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes and newly created branches
    """

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

    conn_dist_weight = cfg_ding0.get('mv_connect', 'load_area_sat_conn_dist_weight')
    conn_dist_ring_mod = cfg_ding0.get('mv_connect', 'load_area_stat_conn_dist_ring_mod')

    for lv_load_area in mv_grid_district.lv_load_areas():

        # exclude aggregated Load Areas and choose only load areas that were connected to grid before
        if not lv_load_area.is_aggregated and \
           lv_load_area.lv_load_area_centre not in mv_grid_district.mv_grid.graph_isolated_nodes():

            lv_load_area_centre = lv_load_area.lv_load_area_centre

            # there's only one station: Replace Load Area centre by station in graph
            if lv_load_area.lv_grid_districts_count() == 1:
                # get station
                lv_station = list(lv_load_area.lv_grid_districts())[0].lv_grid.station()

                # get branches that are connected to Load Area centre
                branches = mv_grid_district.mv_grid.graph_branches_from_node(lv_load_area_centre)

                # connect LV station, delete Load Area centre
                for node, branch in branches:
                    # backup kind and type of branch
                    branch_kind = branch['branch'].kind
                    branch_type = branch['branch'].type
                    branch_ring = branch['branch'].ring

                    # respect circuit breaker if existent
                    circ_breaker = branch['branch'].circuit_breaker
                    if circ_breaker is not None:
                        branch['branch'].circuit_breaker.geo_data = calc_geo_centre_point(lv_station, node)

                    # delete old branch to Load Area centre and create a new one to LV station
                    graph.remove_edge(lv_load_area_centre, node)

                    branch_length = calc_geo_dist_vincenty(lv_station, node)
                    branch = BranchDing0(length=branch_length,
                                         circuit_breaker=circ_breaker,
                                         kind=branch_kind,
                                         type=branch_type,
                                         ring=branch_ring)
                    if circ_breaker is not None:
                        circ_breaker.branch = branch
                    graph.add_edge(lv_station, node, branch=branch)

                # delete Load Area centre from graph
                graph.remove_node(lv_load_area_centre)

            # there're more than one station: Do normal connection process (as in satellites)
            else:
                # connect LV stations of all grid districts
                # =========================================
                for lv_grid_district in lv_load_area.lv_grid_districts():
                    # get branches that are partly or fully located in load area
                    branches = calc_geo_branches_in_polygon(mv_grid_district.mv_grid,
                                                            lv_load_area.geo_area,
                                                            mode='intersects',
                                                            proj=proj1)

                    # filter branches that belong to satellites (load area groups) if Load Area is not a satellite
                    # itself
                    if not lv_load_area.is_satellite:
                        branches_valid = []
                        for branch in branches:
                            node1 = branch['adj_nodes'][0]
                            node2 = branch['adj_nodes'][1]
                            lv_load_area_group = get_lv_load_area_group_from_node_pair(node1, node2)

                            # delete branch as possible conn. target if it belongs to a group (=satellite) or
                            # if it belongs to a ring different from the ring of the current LVLA
                            if (lv_load_area_group is None) and\
                               (branch['branch'].ring is lv_load_area.ring):
                                branches_valid.append(branch)
                        branches = branches_valid

                    # find possible connection objects
                    lv_station = lv_grid_district.lv_grid.station()
                    lv_station_shp = transform(proj1, lv_station.geo_data)
                    conn_objects_min_stack = find_nearest_conn_objects(lv_station_shp, branches, proj1,
                                                                       conn_dist_weight, debug,
                                                                       branches_only=False)

                    # connect!
                    connect_node(lv_station,
                                 lv_station_shp,
                                 mv_grid_district.mv_grid,
                                 conn_objects_min_stack[0],
                                 proj2,
                                 graph,
                                 conn_dist_ring_mod,
                                 debug)

                # Replace Load Area centre by cable distributor
                # ================================================
                # create cable distributor and add it to grid
                cable_dist = MVCableDistributorDing0(geo_data=lv_load_area_centre.geo_data,
                                                     grid=mv_grid_district.mv_grid)
                mv_grid_district.mv_grid.add_cable_distributor(cable_dist)

                # get branches that are connected to Load Area centre
                branches = mv_grid_district.mv_grid.graph_branches_from_node(lv_load_area_centre)

                # connect LV station, delete Load Area centre
                for node, branch in branches:
                    # backup kind and type of branch
                    branch_kind = branch['branch'].kind
                    branch_type = branch['branch'].type
                    branch_ring = branch['branch'].ring

                    # respect circuit breaker if existent
                    circ_breaker = branch['branch'].circuit_breaker
                    if circ_breaker is not None:
                        branch['branch'].circuit_breaker.geo_data = calc_geo_centre_point(cable_dist, node)

                    # delete old branch to Load Area centre and create a new one to LV station
                    graph.remove_edge(lv_load_area_centre, node)

                    branch_length = calc_geo_dist_vincenty(cable_dist, node)
                    branch = BranchDing0(length=branch_length,
                                         circuit_breaker=circ_breaker,
                                         kind=branch_kind,
                                         type=branch_type,
                                         ring=branch_ring)
                    if circ_breaker is not None:
                        circ_breaker.branch = branch
                    graph.add_edge(cable_dist, node, branch=branch)

                # delete Load Area centre from graph
                graph.remove_node(lv_load_area_centre)

            # Replace all overhead lines by cables
            # ====================================
            # if grid's default type is overhead line
            if mv_grid_district.mv_grid.default_branch_kind == 'line':
                # get all branches in load area
                branches = calc_geo_branches_in_polygon(mv_grid_district.mv_grid,
                                                        lv_load_area.geo_area,
                                                        mode='contains',
                                                        proj=proj1)
                # set type
                for branch in branches:
                    branch['branch'].kind = mv_grid_district.mv_grid.default_branch_kind_settle
                    branch['branch'].type = mv_grid_district.mv_grid.default_branch_type_settle

    return graph


def mv_connect_generators(mv_grid_district, graph, debug=False):
    """Connect MV generators to MV grid

    Parameters
    ----------
    mv_grid_district: MVGridDistrictDing0
        MVGridDistrictDing0 object for which the connection process has to be
        done
    graph: :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes
    debug: bool, defaults to False
        If True, information is printed during process.

    Returns
    -------
    :networkx:`NetworkX Graph Obj< >`
        NetworkX graph object with nodes and newly created branches
    """

    generator_buffer_radius = cfg_ding0.get('mv_connect', 'generator_buffer_radius')
    generator_buffer_radius_inc = cfg_ding0.get('mv_connect', 'generator_buffer_radius_inc')

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

    for generator in sorted(mv_grid_district.mv_grid.generators(), key=lambda x: repr(x)):

        # ===== voltage level 4: generator has to be connected to MV station =====
        if generator.v_level == 4:
            mv_station = mv_grid_district.mv_grid.station()

            branch_length = calc_geo_dist_vincenty(generator, mv_station)

            # TODO: set branch type to something reasonable (to be calculated)
            branch_kind = mv_grid_district.mv_grid.default_branch_kind
            branch_type = mv_grid_district.mv_grid.default_branch_type

            branch = BranchDing0(length=branch_length,
                                 kind=branch_kind,
                                 type=branch_type,
                                 ring=None)
            graph.add_edge(generator, mv_station, branch=branch)

            if debug:
                logger.debug('Generator {0} was connected to {1}'.format(
                    generator, mv_station))

        # ===== voltage level 5: generator has to be connected to MV grid (next-neighbor) =====
        elif generator.v_level == 5:
            generator_shp = transform(proj1, generator.geo_data)

            # get branches within a the predefined radius `generator_buffer_radius`
            branches = calc_geo_branches_in_buffer(generator,
                                                   mv_grid_district.mv_grid,
                                                   generator_buffer_radius,
                                                   generator_buffer_radius_inc, proj1)

            # calc distance between generator and grid's lines -> find nearest line
            conn_objects_min_stack = find_nearest_conn_objects(generator_shp,
                                                               branches,
                                                               proj1,
                                                               conn_dist_weight=1,
                                                               debug=debug,
                                                               branches_only=False)

            # connect!
            # go through the stack (from nearest to most far connection target object)
            generator_connected = False
            for dist_min_obj in conn_objects_min_stack:
                # Note 1: conn_dist_ring_mod=0 to avoid re-routing of existent lines
                # Note 2: In connect_node(), the default cable/line type of grid is used. This is reasonable since
                #         the max. allowed power of the smallest possible cable/line type (3.64 MVA for overhead
                #         line of type 48-AL1/8-ST1A) exceeds the max. allowed power of a generator (4.5 MVA (dena))
                #         (if connected separately!)
                target_obj_result = connect_node(generator,
                                                 generator_shp,
                                                 mv_grid_district.mv_grid,
                                                 dist_min_obj,
                                                 proj2,
                                                 graph,
                                                 conn_dist_ring_mod=0,
                                                 debug=debug)

                if target_obj_result is not None:
                    if debug:
                        logger.debug(
                            'Generator {0} was connected to {1}'.format(
                                generator, target_obj_result))
                    generator_connected = True
                    break

            if not generator_connected and debug:
                logger.debug(
                    'Generator {0} could not be connected, try to '
                    'increase the parameter `generator_buffer_radius` in '
                    'config file `config_calc.cfg` to gain more possible '
                    'connection points.'.format(generator))

    return graph
