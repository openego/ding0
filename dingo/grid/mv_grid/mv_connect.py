
from dingo.core.network.stations import *
from dingo.core.network import BranchDingo, CableDistributorDingo

from shapely.geometry import LineString, Point, MultiPoint
from shapely.ops import transform
import pyproj
from functools import partial

import time

def mv_connect(graph, dingo_object, debug=False):
    """ Connects DINGO objects to MV grid, e.g. load areas of type `satellite`, DER etc.

    Method:
        1. Find nearest line for every satellite using shapely distance:
            Transform  to equidistant CRS
        2.

    Args:
        graph: NetworkX graph object with nodes
        dingo_object: component (Dingo object) to be connected
            Valid objects:  LVStationDingo (small load areas that are not incorporated in cvrp MV routing)
                            MVDEA (MV renewable energy plants) (not existent yet)
        debug: If True, information is printed during process

    Returns:
        graph: NetworkX graph object with nodes and newly created branches
    """
    start = time.time()

    # TODO: Move constant to config
    station_dist_thres = 0.75

    # check if dingo_object is valid object
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

        for node in graph.nodes():
            if isinstance(dingo_object, LVStationDingo):

                # station is LV station
                if isinstance(node, LVStationDingo):

                    # satellites only
                    if node.grid.region.is_satellite:

                        satellite_shp = Point(node.geo_data.x, node.geo_data.y)
                        satellite_shp = transform(proj1, satellite_shp)
                        dist_min = 10**6  # initial distance value in m

                        # calc distance between node and grid's lines -> find nearest line
                        # TODO: don't calc distance for all edges, only surrounding ones (but how?)
                        for branch in node.grid.region.mv_region.mv_grid.graph_edges():
                            stations = branch['adj_nodes']

                            # objects for stations and line
                            station1_shp = Point((stations[0].geo_data.x, stations[0].geo_data.y))
                            station2_shp = Point((stations[1].geo_data.x, stations[1].geo_data.y))
                            line_shp = LineString([station1_shp, station2_shp])

                            # transform
                            line_shp = transform(proj1, line_shp)
                            station1_shp = transform(proj1, station1_shp)
                            station2_shp = transform(proj1, station2_shp)

                            # calc distances to stations and line
                            dist = {line_shp: satellite_shp.distance(line_shp),
                                    station1_shp: satellite_shp.distance(station1_shp) * station_dist_thres,
                                    station2_shp: satellite_shp.distance(station2_shp) * station_dist_thres}

                            # # TODO: to be completed
                            # # create dict with possible connection points (objects)
                            # conn_objects = {stations[0]: {'shp': transform(proj1, station1_shp)},
                            #                 stations[1]: {'shp': transform(proj1, station2_shp)},
                            #                 branch: {'shp': transform(proj1, LineString([station1_shp, station2_shp]))}}

                            if min(dist.values()) < dist_min:
                                dist_min = min(dist.values())
                                dist_min_obj = dist
                                branch_dist_min = branch
                                #station_dist_min = line[0]
                                line_shp_dist_min = line_shp

                        # MV line is nearest connection point
                        if isinstance(min(dist_min_obj, key=dist_min_obj.get), LineString):
                            # find nearest point on MV line
                            conn_point_shp = line_shp_dist_min.interpolate(line_shp_dist_min.project(satellite_shp))
                            conn_point_shp = transform(proj2, conn_point_shp)

                            # create cable distributor and add it to grid
                            cable_dist = CableDistributorDingo(geo_data=conn_point_shp)
                            node.grid.region.mv_region.mv_grid.add_cable_distributor(cable_dist)

                            # split old branch into 2 segments (delete old branch and create 2 new ones along cable_dist)
                            graph.remove_edge(branch_dist_min['adj_nodes'][0], branch_dist_min['adj_nodes'][1])
                            graph.add_edge(branch_dist_min['adj_nodes'][0], cable_dist, branch=BranchDingo())
                            graph.add_edge(branch_dist_min['adj_nodes'][1], cable_dist, branch=BranchDingo())

                            # add new branch for satellite (station to cable distributor)
                            graph.add_edge(node, cable_dist, branch=BranchDingo())

                        # MV/LV station ist nearest connection point
                        else:
                            # add new branch for satellite (station to station)
                            graph.add_edge(node, dist_min_obj, branch=BranchDingo())
                            #graph.add_edge(node, branch_dist_min['adj_nodes'][1], branch=BranchDingo())


                        # line = branch_dist_min['adj_nodes']
                        # print('==================================')
                        # print('dist=', dist_min, 'm')
                        # #if dist_min < 100:
                        # #    print('NAH DRAN, DUDE!')
                        # print('Conn. point:', conn_point_shp)
                        # print('Station 1:', line[0].geo_data)
                        # print('Station 2:', line[1].geo_data)
                        # station1 = Point((line[0].geo_data.x, line[0].geo_data.y))
                        # station1 = transform(proj1, station1)
                        # station2 = Point((line[1].geo_data.x, line[1].geo_data.y))
                        # station2 = transform(proj1, station2)
                        # print('Dist. to station:', line_shp_dist_min.interpolate(line_shp_dist_min.project(satellite_shp)).distance(station1), 'm')
                        # print('Dist. to station:', line_shp_dist_min.interpolate(line_shp_dist_min.project(satellite_shp)).distance(station2), 'm')

                        #

                        # TODO: Parametrize new lines!


        print('Elapsed time (mv_connect): {}'.format(time.time() - start))
        return graph

    else:
        print('argument `node` has invalid value, see method for valid inputs.')