
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo
from dingo.core import MVCableDistributorDingo
from dingo.core.structure.groups import LoadAreaGroupDingo
from dingo.core.structure.regions import LVLoadAreaCentreDingo
from dingo.tools import config as cfg_dingo
from dingo.tools.geo import calc_geo_branches_in_buffer,calc_geo_dist_vincenty,\
                            calc_geo_centre_point, calc_geo_branches_in_polygon

from shapely.geometry import LineString
from shapely.ops import transform
import pyproj
from functools import partial

import time


def find_nearest_conn_objects(node_shp, branches, proj, conn_dist_weight, debug, branches_only=False):
    """ Searches all `branches` for the nearest possible connection object per branch (picks out 1 object out of 3
        possible objects: 2 branch-adjacent stations and 1 potentially created cable distributor on the line
        (perpendicular projection)). The resulting stack (list) is sorted ascending by distance from node.

    Args:
        node_shp: Shapely Point object of node
        branches: BranchDingo objects of MV region
        proj: pyproj projection object: nodes' CRS to equidistant CRS (e.g. WGS84 -> ETRS)
        conn_dist_weight: length weighting to prefer stations instead of direct line connection,
                          see mv_connect_satellites() for details.
        debug: If True, information is printed during process
        branches_only: If True, only branch objects are considered as connection objects

    Returns:
        conn_objects_min_stack: List of connection objects (each object is represented by dict with Dingo object,
                                shapely object and distance to node.

    """

    conn_objects_min_stack = []

    for branch in branches:
        stations = branch['adj_nodes']

        # create shapely objects for 2 stations and line between them, transform to equidistant CRS
        station1_shp = transform(proj, stations[0].geo_data)
        station2_shp = transform(proj, stations[1].geo_data)
        line_shp = LineString([station1_shp, station2_shp])

        # create dict with DINGO objects (line & 2 adjacent stations), shapely objects and distances
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

            # remove MV station as possible connection point
            if isinstance(conn_objects['s1']['obj'], MVStationDingo):
                del conn_objects['s1']
            elif isinstance(conn_objects['s2']['obj'], MVStationDingo):
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
        print('Stack length:', len(conn_objects_min_stack))

    return conn_objects_min_stack


def get_lv_load_area_group_from_node_pair(node1, node2):

    lv_load_area_group = None

    # both nodes are LV stations -> get group from 1 or 2
    if (isinstance(node1, LVLoadAreaCentreDingo) and
       isinstance(node2, LVLoadAreaCentreDingo)):
        if not node1.lv_load_area.lv_load_area_group:
            lv_load_area_group = node2.lv_load_area.lv_load_area_group
        else:
            lv_load_area_group = node1.lv_load_area.lv_load_area_group

    # node 1 is LV station and node 2 not -> get group from node 1
    elif (isinstance(node1, LVLoadAreaCentreDingo) and
          isinstance(node2, (MVStationDingo, MVCableDistributorDingo))):
        lv_load_area_group = node1.lv_load_area.lv_load_area_group

    # node 2 is LV station and node 1 not -> get group from node 2
    elif (isinstance(node1, (MVStationDingo, MVCableDistributorDingo)) and
          isinstance(node2, LVLoadAreaCentreDingo)):
        lv_load_area_group = node2.lv_load_area.lv_load_area_group

    # both nodes are not a LV station -> no group
    elif (isinstance(node1, (MVStationDingo, MVCableDistributorDingo)) and
          isinstance(node2, (MVStationDingo, MVCableDistributorDingo))):
        lv_load_area_group = None

    return lv_load_area_group


def find_connection_point(node, node_shp, graph, proj, conn_objects_min_stack, conn_dist_ring_mod, debug):
    """ Goes through the possible target connection objects in `conn_objects_min_stack` (from nearest to most far
        object) and tries to connect `node` to one of them.

    Args:
        node: origin node - Dingo object (e.g. LVLoadAreaCentreDingo)
        node_shp: Shapely Point object of node
        graph: NetworkX graph object with nodes
        proj: pyproj projection object: equidistant CRS to conformal CRS (e.g. ETRS -> WGS84)
        conn_objects_min_stack: List of connection objects (each object is represented by dict with Dingo object,
                                shapely object and distance to node), sorted ascending by distance.
        conn_dist_ring_mod: Max. distance when nodes are included into route instead of creating a new line,
                            see mv_connect() for details.
        debug: If True, information is printed during process

    Returns:
        nothing
    """

    node_connected = False

    # go through the stack (from nearest to most far connection target object)
    for dist_min_obj in conn_objects_min_stack:

        # target object is branch
        if isinstance(dist_min_obj['shp'], LineString):
            # rename for readability
            node1 = dist_min_obj['obj']['adj_nodes'][0]
            node2 = dist_min_obj['obj']['adj_nodes'][1]

            lv_load_area_group = get_lv_load_area_group_from_node_pair(node1, node2)

        # target object is node
        else:
            if isinstance(dist_min_obj['obj'], MVCableDistributorDingo):
                lv_load_area_group = dist_min_obj['obj'].lv_load_area_group
            else:
                lv_load_area_group = dist_min_obj['obj'].lv_load_area.lv_load_area_group

        # target object doesn't belong to a satellite string (is member of a LV load_area group)
        if not lv_load_area_group:

            # connect node
            target_obj_result = connect_node(node, node_shp, dist_min_obj, proj, graph, conn_dist_ring_mod, debug)

            # if node was connected via branch (target line not re-routed and not member of aggregated load area):
            # create new LV load_area group for current node
            if (target_obj_result is not None) and (target_obj_result != 're-routed'):
                lv_load_area_group = LoadAreaGroupDingo(mv_grid_district=node.lv_load_area.mv_grid_district,
                                                        root_node=target_obj_result)
                lv_load_area_group.add_lv_load_area(lv_load_area=node.lv_load_area)
                node.lv_load_area.lv_load_area_group = lv_load_area_group
                node.lv_load_area.mv_grid_district.add_lv_load_area_group(lv_load_area_group)

                if debug:
                    print('New LV load_area group', lv_load_area_group, 'created!')

                # node connected, stop connection for current node
                node_connected = True
                break

            # node was inserted into line (target line was re-routed)
            elif target_obj_result == 're-routed':
                # node connected, stop connection for current node
                node_connected = True
                break

        # target object is member of a LV load_area group
        else:

            # connect node
            target_obj_result = connect_node(node, node_shp, dist_min_obj, proj, graph, conn_dist_ring_mod, debug)

            # if node was connected via branch (target line not re-routed and not member of aggregated load area):
            # create new LV load_area group for current node
            if (target_obj_result is not None) and (target_obj_result != 're-routed'):
                # node can join LV load_area group
                if lv_load_area_group.can_add_lv_load_area(node=node):

                    # add node to LV load_area group
                    lv_load_area_group.add_lv_load_area(lv_load_area=node.lv_load_area)
                    node.lv_load_area.lv_load_area_group = lv_load_area_group

                    if isinstance(target_obj_result, MVCableDistributorDingo):
                        lv_load_area_group.add_lv_load_area(lv_load_area=target_obj_result)
                        target_obj_result.lv_load_area_group = lv_load_area_group

                    if debug:
                        print('LV load_area group', lv_load_area_group, 'joined!')

                    # node connected, stop connection for current node
                    node_connected = True
                    break

                # cannot join LV load_area group
                else:
                    if debug:
                        print('Node', node, 'could not be added to load_area group', lv_load_area_group)

                    # rollback changes in graph
                    disconnect_node(node, target_obj_result, graph, debug)

                    # continue with next possible connection point
                    continue

            # node was inserted into line (target line was re-routed)
            elif target_obj_result == 're-routed':
                # add node to LV load_area group
                lv_load_area_group.add_lv_load_area(lv_load_area=node.lv_load_area)
                node.lv_load_area.lv_load_area_group = lv_load_area_group
                node.lv_load_area.is_satellite = False

                # node inserted into existing route, stop connection for current node
                node_connected = True
                break

            # else: node could not be connected because target object belongs to load area of aggregated type

    if not node_connected:
        print('Node', node, 'could not be connected, try to increase the parameter `load_area_sat_buffer_radius` in',
              'config file `config_calc` to gain more possible connection points.')


def connect_node(node, node_shp, target_obj, proj, graph, conn_dist_ring_mod, debug):
    """ Connects `node` to `target_obj`

    Args:
        node: origin node - Dingo object (e.g. LVLoadAreaCentreDingo)
        node_shp: Shapely Point object of origin node
        target_obj: object that node shall be connected to
        proj: pyproj projection object: equidistant CRS to conformal CRS (e.g. ETRS -> WGS84)
        graph: NetworkX graph object with nodes and newly created branches
        conn_dist_ring_mod: Max. distance when nodes are included into route instead of creating a new line,
                            see mv_connect() for details.
        debug: If True, information is printed during process

    Returns:
        target_obj_result: object that node was connected to (instance of LVLoadAreaCentreDingo or
                           MVCableDistributorDingo). If node is included into line instead of creating a new line (see arg
                           `conn_dist_ring_mod`), `target_obj_result` is None.
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
                # backup type of branch
                branch_type = graph.edge[adj_node1][adj_node2]['branch'].type

                # check if there's a circuit breaker on current branch,
                # if yes set new position between first node (adj_node1) and newly inserted node
                circ_breaker = graph.edge[adj_node1][adj_node2]['branch'].circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = calc_geo_centre_point(adj_node1, node)

                # split old ring main route into 2 segments (delete old branch and create 2 new ones
                # along node)
                graph.remove_edge(adj_node1, adj_node2)

                branch_length = calc_geo_dist_vincenty(adj_node1, node)
                branch = BranchDingo(length=branch_length,
                                     circuit_breaker=circ_breaker,
                                     type=branch_type)
                if circ_breaker is not None:
                    circ_breaker.branch = branch
                graph.add_edge(adj_node1, node, branch=branch)

                branch_length = calc_geo_dist_vincenty(adj_node2, node)
                graph.add_edge(adj_node2, node, branch=BranchDingo(length=branch_length,
                                                                   type=branch_type))

                target_obj_result = 're-routed'

                if debug:
                    print('Ring main route modified to include node', node)

            # Node is too far away from route
            # => keep main route and create new line from node to (cable distributor on) route.
            else:

                # create cable distributor and add it to grid
                cable_dist = MVCableDistributorDingo(geo_data=conn_point_shp,
                                                     grid=node.lv_load_area.mv_grid_district.mv_grid)
                node.lv_load_area.mv_grid_district.mv_grid.add_cable_distributor(cable_dist)

                # check if there's a circuit breaker on current branch,
                # if yes set new position between first node (adj_node1) and newly created cable distributor
                circ_breaker = graph.edge[adj_node1][adj_node2]['branch'].circuit_breaker
                if circ_breaker is not None:
                    circ_breaker.geo_data = calc_geo_centre_point(adj_node1, cable_dist)

                # split old branch into 2 segments (delete old branch and create 2 new ones along cable_dist)
                # ===========================================================================================

                # backup type of branch
                branch_type = graph.edge[adj_node1][adj_node2]['branch'].type

                graph.remove_edge(adj_node1, adj_node2)

                branch_length = calc_geo_dist_vincenty(adj_node1, cable_dist)
                branch = BranchDingo(length=branch_length,
                                     circuit_breaker=circ_breaker,
                                     type=branch_type)
                if circ_breaker is not None:
                    circ_breaker.branch = branch
                graph.add_edge(adj_node1, cable_dist, branch=branch)

                branch_length = calc_geo_dist_vincenty(adj_node2, cable_dist)
                graph.add_edge(adj_node2, cable_dist, branch=BranchDingo(length=branch_length,
                                                                         type=type))

                # add new branch for satellite (station to cable distributor)
                # ===========================================================

                # get default branch type from grid to use it for new branch
                branch_type = node.lv_load_area.mv_grid_district.mv_grid.default_branch_type

                branch_length = calc_geo_dist_vincenty(node, cable_dist)
                graph.add_edge(node, cable_dist, branch=BranchDingo(length=branch_length,
                                                                    type=branch_type))
                target_obj_result = cable_dist

                # debug info
                if debug:
                    print('Nearest connection point for object', node, 'is branch',
                          target_obj['obj']['adj_nodes'], '(distance=', target_obj['dist'], 'm)')

    # node ist nearest connection point
    else:

        # target node is not a load area of type aggregated
        if isinstance(target_obj['obj'], LVLoadAreaCentreDingo) and not target_obj['obj'].lv_load_area.is_aggregated:

            # get default branch type from grid to use it for new branch
            branch_type = node.lv_load_area.mv_grid_district.mv_grid.default_branch_type

            # add new branch for satellite (station to station)
            branch_length = calc_geo_dist_vincenty(node, target_obj['obj'])
            graph.add_edge(node, target_obj['obj'], branch=BranchDingo(length=branch_length,
                                                                       type=branch_type))
            target_obj_result = target_obj['obj']

            # debug info
            if debug:
                print('Nearest connection point for object', node, 'is station',
                      target_obj['obj'], '(distance=', target_obj['dist'], 'm)')

    return target_obj_result


def disconnect_node(node, target_obj_result, graph, debug):
    """ Disconnects `node` from `target_obj`

    Args:
        node: node - Dingo object (e.g. LVLoadAreaCentreDingo)
        target_obj_result:
        graph: NetworkX graph object with nodes and newly created branches
        debug: If True, information is printed during process

    Returns:
        nothing
    """

    # backup type of branch
    branch_type = graph.edge[node][target_obj_result]['branch'].type

    graph.remove_edge(node, target_obj_result)

    if isinstance(target_obj_result, MVCableDistributorDingo):

        neighbor_nodes = graph.neighbors(target_obj_result)

        if len(neighbor_nodes) == 2:
            graph.remove_node(target_obj_result)

            branch_length = calc_geo_dist_vincenty(neighbor_nodes[0], neighbor_nodes[1])
            graph.add_edge(neighbor_nodes[0], neighbor_nodes[1], branch=BranchDingo(length=branch_length,
                                                                                    type=branch_type))

    if debug:
        print('disconnect edge', node, '-', target_obj_result)


def parametrize_lines(mv_grid):
    """ Set unparametrized branches to default branch type
    Args:
        mv_grid: MVGridDingo object

    Returns:
        nothing

    Notes:
        During the connection process of satellites, new branches are created - these have to be parametrized.
    """

    for branch in mv_grid.graph_edges():
        if branch['branch'].type is None:
            branch['branch'].type = mv_grid.default_branch_type


# TODO: MAKE MV_CONNECT THE START METHOD FOR CONNECTION, SPLIT TYPES INTO SUBMETHODS
def mv_connect_satellites(mv_grid, graph, dingo_object, debug=False):
    """ Connects DINGO objects to MV grid, e.g. load areas of type `satellite`, DER etc.

    Method:
        1. Find nearest line for every satellite using shapely distance:
            Transform  to equidistant CRS
        2. ...

    Args:
        mv_grid: MVGridDingo object
        graph: NetworkX graph object with nodes
        dingo_object: component (instance(!) of Dingo class) to be connected
            Valid objects:  LVStationDingo (small load areas that are not incorporated in cvrp MV routing)
                            MVDEA (MV renewable energy plants) (not existent yet)
            CAUTION: `dingo_object` is not connected but it specifies the types of objects that are to be connected,
                     e.g. if LVStationDingo() is passed, all objects of this type within `graph` are connected.
        debug: If True, information is printed during process

    Returns:
        graph: NetworkX graph object with nodes and newly created branches
    """

    # conn_dist_weight: The satellites can be connected to line (new terminal is created) or to one station where the
    # line ends, depending on the distance from satellite to the objects. This threshold is a length weighting to prefer
    # stations instead of direct line connection to respect grid planning principles.
    # Example: The distance from satellite to line is 1km, to station1 1.2km, to station2 2km.
    # With conn_dist_threshold=0.75, the 'virtual' distance to station1 would be 1.2km * 0.75 = 0.9km, so this conn.
    # point would be preferred.
    conn_dist_weight = cfg_dingo.get('mv_connect', 'load_area_sat_conn_dist_weight')

    # conn_dist_ring_mod: Allow re-routing of ring main route if node is closer than this threshold (in m) to ring.
    conn_dist_ring_mod = cfg_dingo.get('mv_connect', 'load_area_sat_conn_dist_ring_mod')

    load_area_sat_buffer_radius = cfg_dingo.get('mv_connect', 'load_area_sat_buffer_radius')
    load_area_sat_buffer_radius_inc = cfg_dingo.get('mv_connect', 'load_area_sat_buffer_radius_inc')

    start = time.time()

    # check if dingo_object is valid object (can be connected)
    # TODO: Add RES to isinstance check
    if isinstance(dingo_object, (LVLoadAreaCentreDingo, LVStationDingo)):

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


        # TODO: create generators in grid class for iterating over satellites and non-satellites (nice-to-have) instead
        # TODO: of iterating over all nodes
        # check all nodes
        for node in sorted(graph.nodes(), key=lambda x: repr(x)):
            # LV load area centres are to be connected (= satellite handling)
            if isinstance(dingo_object, LVLoadAreaCentreDingo):

                # node is LV load area centre
                if isinstance(node, LVLoadAreaCentreDingo):

                    # satellites only
                    if node.lv_load_area.is_satellite:

                        node_shp = transform(proj1, node.geo_data)

                        branches = calc_geo_branches_in_buffer(node,
                                                               load_area_sat_buffer_radius,
                                                               load_area_sat_buffer_radius_inc, proj1)

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
            print('Elapsed time (mv_connect): {}'.format(time.time() - start))

        return graph

    else:
        print('argument `dingo_object` has invalid value, see method for valid inputs.')


def mv_connect_stations(mv_grid_district, graph, debug=False):

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

    conn_dist_weight = cfg_dingo.get('mv_connect', 'load_area_sat_conn_dist_weight')
    conn_dist_ring_mod = cfg_dingo.get('mv_connect', 'load_area_stat_conn_dist_ring_mod')

    # for every station: find nearest branch and connect to it

    # set branch in load area to cable type (type and start- and end point)

    for lv_load_area in mv_grid_district.lv_load_areas():

        # exclude aggregated LV load areas and choose only load areas that were connected to grid before
        if not lv_load_area.is_aggregated and lv_load_area.is_connected():

            # ===== DEBUG STUFF (BUG JONAS) =====
            # TODO: Remove when fixed!
            if lv_load_area.lv_grid_districts_count() == 0:
                print('No station for', lv_load_area, 'found! (blame Jonas)')
            # ===================================

            lv_load_area_centre = lv_load_area.lv_load_area_centre

            # there's only one station: Replace LV load area centre by station in graph
            if lv_load_area.lv_grid_districts_count() == 1:
                # get station
                lv_station = list(lv_load_area.lv_grid_districts())[0].lv_grid.station()

                # get branches that are connected to LV load area centre
                branches = mv_grid_district.mv_grid.graph_branches_from_node(lv_load_area_centre)

                # connect LV station, delete LV load area centre
                for node, branch in branches:
                    # backup type of branch
                    branch_type = branch['branch'].type

                    # respect circuit breaker if existent
                    circ_breaker = branch['branch'].circuit_breaker
                    if circ_breaker is not None:
                        branch['branch'].circuit_breaker.geo_data = calc_geo_centre_point(lv_station, node)

                    # delete old branch to LV load area centre and create a new one to LV station
                    graph.remove_edge(lv_load_area_centre, node)

                    branch_length = calc_geo_dist_vincenty(lv_station, node)
                    branch = BranchDingo(length=branch_length,
                                         circuit_breaker=circ_breaker,
                                         type=branch_type)
                    if circ_breaker is not None:
                        circ_breaker.branch = branch
                    graph.add_edge(lv_station, node, branch=branch)

                # delete LV load area centre from graph
                graph.remove_node(lv_load_area_centre)

            # there're more than one station: Do normal connection process (as in satellites)
            else:
                # connect LV stations of all grid districts
                # =========================================
                for lv_grid_district in lv_load_area.lv_grid_districts():
                    # get branches that are partly or fully located in load area
                    branches = calc_geo_branches_in_polygon(mv_grid_district.mv_grid, lv_load_area.geo_area, proj1)

                    # filter branches that belong to satellites (load area groups) if LV load area is not a satellite
                    # itself
                    if not lv_load_area.is_satellite:
                        for branch in branches:
                            node1 = branch['adj_nodes'][0]
                            node2 = branch['adj_nodes'][1]
                            lv_load_area_group = get_lv_load_area_group_from_node_pair(node1, node2)

                            if lv_load_area_group is not None:
                                branches.remove(branch)

                    # find possible connection objects
                    lv_station = lv_grid_district.lv_grid.station()
                    lv_station_shp = transform(proj1, lv_station.geo_data)
                    conn_objects_min_stack = find_nearest_conn_objects(lv_station_shp, branches, proj1,
                                                                       conn_dist_weight, debug,
                                                                       branches_only=False)

                    # connect!
                    connect_node(lv_station, lv_station_shp, conn_objects_min_stack[0], proj2,
                                 graph, conn_dist_ring_mod, debug)

                # Replace LV load area centre by cable distributor
                # ================================================
                # create cable distributor and add it to grid
                cable_dist = MVCableDistributorDingo(geo_data=lv_load_area_centre.geo_data,
                                                     grid=mv_grid_district.mv_grid)
                mv_grid_district.mv_grid.add_cable_distributor(cable_dist)

                # get branches that are connected to LV load area centre
                branches = mv_grid_district.mv_grid.graph_branches_from_node(lv_load_area_centre)

                # connect LV station, delete LV load area centre
                for node, branch in branches:
                    # backup type of branch
                    branch_type = branch['branch'].type

                    # respect circuit breaker if existent
                    circ_breaker = branch['branch'].circuit_breaker
                    if circ_breaker is not None:
                        branch['branch'].circuit_breaker.geo_data = calc_geo_centre_point(cable_dist, node)

                    # delete old branch to LV load area centre and create a new one to LV station
                    graph.remove_edge(lv_load_area_centre, node)

                    branch_length = calc_geo_dist_vincenty(cable_dist, node)
                    branch = BranchDingo(length=branch_length,
                                         circuit_breaker=circ_breaker,
                                         type=branch_type)
                    if circ_breaker is not None:
                        circ_breaker.branch = branch
                    graph.add_edge(cable_dist, node, branch=branch)

                # delete LV load area centre from graph
                graph.remove_node(lv_load_area_centre)
