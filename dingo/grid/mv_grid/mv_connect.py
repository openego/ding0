
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo, CableDistributorDingo
from dingo.core.structure.groups import LVRegionGroupDingo
from dingo.tools import config as cfg_dingo
from dingo.tools.geo import calc_geo_branches_in_buffer

from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import transform
import pyproj
from functools import partial

import time


def find_nearest_conn_objects(node_shp, branches, proj, conn_dist_weight, debug):
    """ Searches all `branches` for the nearest possible connection object per branch (picks out 1 object out of 3
        possible objects: 2 branch-adjacent stations and 1 potentially created cable distributor on the line
        (perpendicular projection)). The resulting stack (list) is sorted ascending by distance from node.

    Args:
        node_shp:
        branches:
        proj:
        conn_dist_weight:
        debug:

    Returns:

    """

    conn_objects_min_stack = []

    for branch in branches:
        stations = branch['adj_nodes']

        # cshapely objects for 2 stations and line between them, transform to equidistant CRS
        station1_shp = transform(proj, stations[0].geo_data)
        station2_shp = transform(proj, stations[1].geo_data)
        line_shp = LineString([station1_shp, station2_shp])

        # create dict with DINGO objects (line & 2 adjacent stations), shapely objects and distances
        conn_objects = {'s1': {'obj': stations[0],
                               'shp': station1_shp,
                               'dist': node_shp.distance(station1_shp) * conn_dist_weight},
                        's2': {'obj': stations[1],
                               'shp': station2_shp,
                               'dist': node_shp.distance(station2_shp) * conn_dist_weight},
                        'b': {'obj': branch,
                              'shp': line_shp,
                              'dist': node_shp.distance(line_shp)}}

        # remove MV station as possible connection point
        if isinstance(conn_objects['s1']['obj'], MVStationDingo):
            del conn_objects['s1']
        elif isinstance(conn_objects['s2']['obj'], MVStationDingo):
            del conn_objects['s2']

        # find nearest connection point on given triple dict (2 branch-adjacent stations + cable dist. on line)
        conn_objects_min = min(conn_objects.values(), key=lambda v: v['dist'])
        conn_objects_min_stack.append(conn_objects_min)

    # sort all objects by distance from node
    conn_objects_min_stack = [_ for _ in sorted(conn_objects_min_stack, key=lambda x: x['dist'])]

    if debug:
        print('Stack length:', len(conn_objects_min_stack))

    return conn_objects_min_stack


def find_connection_point(node, node_shp, graph, proj, conn_objects_min_stack, conn_dist_ring_mod, debug):
    """

    Args:
        node:
        node_shp:
        graph:
        proj:
        conn_objects_min_stack:
        conn_dist_ring_mod:
        debug:

    Returns:

    """

    for dist_min_obj in conn_objects_min_stack:

        # target object is branch
        if isinstance(dist_min_obj['shp'], LineString):
            # rename for readability
            node1 = dist_min_obj['obj']['adj_nodes'][0]
            node2 = dist_min_obj['obj']['adj_nodes'][1]

            # both nodes are LV stations -> get group from 1 or 2
            if (isinstance(node1, LVStationDingo) and
               isinstance(node2, LVStationDingo)):
                if not node1.grid.region.lv_region_group:
                    lv_region_group = node2.grid.region.lv_region_group
                else:
                    lv_region_group = node1.grid.region.lv_region_group

            # node 1 is LV station and node 2 not -> get group from node 1
            elif (isinstance(node1, LVStationDingo) and
                  isinstance(node2, (MVStationDingo, CableDistributorDingo))):
                lv_region_group = node1.grid.region.lv_region_group

            # node 2 is LV station and node 1 not -> get group from node 2
            elif (isinstance(node1, (MVStationDingo, CableDistributorDingo)) and
                  isinstance(node2, LVStationDingo)):
                lv_region_group = node2.grid.region.lv_region_group

            # both nodes are not a LV station -> no group
            elif (isinstance(node1, (MVStationDingo, CableDistributorDingo)) and
                  isinstance(node2, (MVStationDingo, CableDistributorDingo))):
                lv_region_group = None

        # target object is node
        else:
            if not isinstance(dist_min_obj['obj'], CableDistributorDingo):
                lv_region_group = dist_min_obj['obj'].grid.region.lv_region_group
            else:
                lv_region_group = None

        # target object doesn't belong to a satellite string (is member of a LV region group)
        if lv_region_group is None:

            branch_created = connect_node(node, node_shp, dist_min_obj, proj, graph, conn_dist_ring_mod, debug)

            # if node was connected via branch: create new LV region group for current node
            if branch_created:
                lv_region_group = LVRegionGroupDingo(id_db=node.grid.region.mv_region.lv_region_groups_count() + 1)
                lv_region_group.add_lv_region(node.grid.region, dist_min_obj['dist'])
                node.grid.region.lv_region_group = lv_region_group
                node.grid.region.mv_region.add_lv_region_group(lv_region_group)

            if debug:
                print('New LV region created!')
            break

        # target object is member of a LV region group
        else:
            if lv_region_group.can_add_lv_region(lv_region=node.grid.region, branch_length=dist_min_obj['dist']):

                branch_created = connect_node(node, node_shp, dist_min_obj, proj, graph, conn_dist_ring_mod, debug)

                # if node was connected via branch: create new LV region group for current node
                if branch_created:
                    lv_region_group.add_lv_region(node.grid.region, dist_min_obj['dist'])
                    node.grid.region.lv_region_group = lv_region_group

                if debug:
                    print('LV region joined!')
                break
            else:
                if debug:
                    print('Node', node, 'could not be added to group:')
                    print('distance(node)=', dist_min_obj['dist'], ', distance_sum(group)=', lv_region_group.branch_length_sum)
                    print('load(node)=', node.grid.region.peak_load_sum, ', load_sum(group)=', lv_region_group.peak_load_sum)
                continue


def connect_node(node, node_shp, dist_min_obj, proj, graph, conn_dist_ring_mod, debug):
    """
    Args:
        node: origin node
        node_shp:
        dist_min_obj:
        proj:
        graph:
        conn_dist_ring_mod:
        debug:

    Returns:
        branch_length: length of newly created branch
    """

    branch_created = False

    # MV line is nearest connection point
    if isinstance(dist_min_obj['shp'], LineString):

        # find nearest point on MV line
        conn_point_shp = dist_min_obj['shp'].interpolate(dist_min_obj['shp'].project(node_shp))
        conn_point_shp = transform(proj, conn_point_shp)

        # Node is close to line
        # -> insert node into route (change existing route)
        if dist_min_obj['dist'] < conn_dist_ring_mod:

            # split old ring main route into 2 segments (delete old branch and create 2 new ones
            # along node)
            graph.remove_edge(dist_min_obj['obj']['adj_nodes'][0], dist_min_obj['obj']['adj_nodes'][1])
            graph.add_edge(dist_min_obj['obj']['adj_nodes'][0], node, branch=BranchDingo())
            graph.add_edge(dist_min_obj['obj']['adj_nodes'][1], node, branch=BranchDingo())

            if debug:
                print('Ring main route modified to include node', node)

        # Node is too far away from route
        # => keep main route and create new line from node to (cable distributor on) route.
        else:

            # create cable distributor and add it to grid
            cable_dist = CableDistributorDingo(geo_data=conn_point_shp, grid=node.grid)
            node.grid.region.mv_region.mv_grid.add_cable_distributor(cable_dist)

            # split old branch into 2 segments (delete old branch and create 2 new ones along cable_dist)
            graph.remove_edge(dist_min_obj['obj']['adj_nodes'][0], dist_min_obj['obj']['adj_nodes'][1])
            graph.add_edge(dist_min_obj['obj']['adj_nodes'][0], cable_dist, branch=BranchDingo())
            graph.add_edge(dist_min_obj['obj']['adj_nodes'][1], cable_dist, branch=BranchDingo())

            # add new branch for satellite (station to cable distributor)
            graph.add_edge(node, cable_dist, branch=BranchDingo())
            branch_created = True

            # debug info
            if debug:
                print('Nearest connection point for object', node, 'is branch',
                      dist_min_obj['obj']['adj_nodes'], '(distance=', dist_min_obj['dist'], 'm)')

    # node ist nearest connection point
    else:
        # add new branch for satellite (station to station)
        graph.add_edge(node, dist_min_obj['obj'], branch=BranchDingo())
        branch_created = True

        # debug info
        if debug:
            print('Nearest connection point for object', node, 'is station',
                  dist_min_obj['obj'], '(distance=', dist_min_obj['dist'], 'm)')

    return branch_created


def mv_connect(graph, dingo_object, debug=False):
    """ Connects DINGO objects to MV grid, e.g. load areas of type `satellite`, DER etc.

    Method:
        1. Find nearest line for every satellite using shapely distance:
            Transform  to equidistant CRS
        2. ...

    Args:
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
    # TODO: Complete docstring

    start = time.time()

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

    # check if dingo_object is valid object
    # TODO: Add RES to isinstance check
    if isinstance(dingo_object, (LVStationDingo, LVStationDingo)):



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
        # TODO: create generators in grid class for iterating over satellites and non-satellites
        for node in graph.nodes():
            if isinstance(dingo_object, LVStationDingo):

                # station is LV station
                if isinstance(node, LVStationDingo):

                    # satellites only
                    if node.grid.region.is_satellite:

                        node_shp = transform(proj1, node.geo_data)

                        branches = calc_geo_branches_in_buffer(node,
                                                               load_area_sat_buffer_radius,
                                                               load_area_sat_buffer_radius_inc, proj1)

                        # === FIND ===
                        # calc distance between node and grid's lines -> find nearest line
                        conn_objects_min_stack = find_nearest_conn_objects(node_shp, branches, proj1, conn_dist_weight, debug)
                        #print('length stack:', len(conn_objects_min_stack))

                        # === iterate over object stack ===
                        find_connection_point(node, node_shp, graph, proj2, conn_objects_min_stack, conn_dist_ring_mod, debug)


                        # TODO: Parametrize new lines!
        print('Elapsed time (mv_connect): {}'.format(time.time() - start))
        if debug:
            print('Elapsed time (mv_connect): {}'.format(time.time() - start))

        return graph

    else:
        print('argument `dingo_object` has invalid value, see method for valid inputs.')
