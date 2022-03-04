from shapely.geometry import MultiPoint, Point, shape, box

import pyproj
import pandas as pd

import numpy as np
from scipy.spatial.distance import pdist, squareform
from ding0.core.network.loads import MVLoadDing0
from ding0.core.network.stations import LVStationDing0



def convertCoords(coordx, coordy, from_proj, to_proj):
    """pyproj.transform from_proj, to_proj"""

    from_proj = pyproj.Proj(init='epsg:'+str(from_proj))
    to_proj   = pyproj.Proj(init='epsg:'+str(to_proj))

    x2,y2 = pyproj.transform(from_proj, to_proj, coordx, coordy)
    return pd.Series([x2, y2])


def get_Point_from_xy(x, y):
    """ 
    get point for given x,y
    """
    return Point(x,y)


def get_points_in_load_area(geos_list):
    
    """ 
    get all points in a load area
    based on buildings, e.g.:
        buildings_with_amenities
        buildings_without_amenities
        amenities_not_in_buildings
    """
    point_lists = []
    
    for geometry in geos_list:

        if geometry.geom_type == 'Polygon':
            point_lists += [Point(point) for point in geometry.exterior.coords[:-1]]

        elif geometry.geom_type == 'Point':

            point_lists.append(geometry)

        else:
            raise IOError('Shape is not a polygon neither a point.')

    return point_lists


def get_bounding_box_from_points(points):
    """
    create bounding_box based on points.
    """
    mpt = MultiPoint([shape(point) for point in points])
    return box(*mpt.bounds, ccw=True)


def get_convex_hull_from_points(points):
    """ return convex hull for given points"""
    
    mpt = MultiPoint([shape(point) for point in points])
    return mpt.convex_hull


def get_load_center(lv_load_area):
    """
    get station which is load center to set its
    geo_data as load center of load areal.
    """

    '''if len(lv_load_area._lv_grid_districts) <= 2:

        logger.warning(lv_load_area._lv_grid_districts)
        station = lv_load_area._lv_grid_districts[0].lv_grid._station

    else:'''

    peak_loads = []
    coordinates = []

    lv_load_area_supply_nodes = list(lvgd.lv_grid.station() for lvgd in lv_load_area.lv_grid_districts()) \
                                + list(lv_load_area._mv_loads)

    for node in lv_load_area_supply_nodes:
        peak_loads.append(node.peak_load)
        coordinates.append(node.geo_data)

    coords = [[p.x, p.y] for p in coordinates]
    coords_array = np.array(coords)
    dist_array = pdist(coords_array)
    dist_matrix = squareform(dist_array)
    unweighted_nodes = dist_matrix.dot(peak_loads)
    load_center_ix = int(np.where(unweighted_nodes == np.amin(unweighted_nodes))[0][0])

    centre = lv_load_area_supply_nodes[load_center_ix]

    # check if load are centre intersects with load area polygon
    # if not extend border such that all supply nodes are inside load area
    if lv_load_area.geo_area.intersects(centre.geo_data):
        load_area_geo = lv_load_area.geo_area
    else:
        points = get_points_in_load_area(coordinates)
        polygon_extended = get_convex_hull_from_points(points)
        load_area_geo = lv_load_area.geo_area.union(polygon_extended)

    if isinstance(centre, LVStationDing0):  # TODO make consistent
        centre_osm = centre.osm_id_node
    elif isinstance(centre, MVLoadDing0):
        centre_osm = centre.osmid_building

    return centre_osm, centre.geo_data, load_area_geo
