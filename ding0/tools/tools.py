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


from geopy import distance
from shapely.geometry import Point, LineString, LinearRing, Polygon


def merge_two_dicts(x, y):
    '''Given two dicts, merge them into a new dict as a shallow copy.
    Parameters
    ----------
    x: dict
    y: dict
    Notes
    -----
    This function was originally proposed by
    http://stackoverflow.com/questions/38987/how-to-merge-two-python-dictionaries-in-a-single-expression
    Credits to Thomas Vander Stichele. Thanks for sharing ideas!
    Returns
    -------
    :obj:`dict`
        Merged dictionary keyed by top-level keys of both dicts
    '''

    z = x.copy()
    z.update(y)
    return z


def get_dest_point(source_point, distance_m, bearing_deg):
    """
    Get the WGS84 point / in the coordinate reference system
    epsg 4326 at a distance (in meters) from
    a source point in a given bearing (in degrees)
    like the `bsdgame trek <https://en.wikipedia.org/wiki/Star_Trek_(1971_video_game)`_
    on linux (0 degrees being North and clockwise is positive).
    Parameters
    ----------
    source_point: :shapely:`Shapely Point object<points>`
        The start point in WGS84 or epsg 4326 coordinates
    distance_m: :obj:`float`
        Distance of destination point from source in meters
    bearing_deg: :obj:`float`
        Bearing of destination point from source in degrees,
        0 degrees being North and clockwise is positive
        (like the `bsdgame trek <https://en.wikipedia.org/wiki/Star_Trek_(1971_video_game)`_
        on linux).
    Returns
    -------
    :shapely:`Shapely Point object<points>`
        The point in WGS84 or epsg 4326 coordinates
        at the destination which is distance meters
        away from the source_point in the bearing provided
    """
    geopy_dest = (distance
                  .distance(meters=distance_m)
                  .destination((source_point.y,
                                source_point.x),
                               bearing_deg))
    return Point(geopy_dest.longitude, geopy_dest.latitude)


def get_cart_dest_point(source_point, east_meters, north_meters):
    """
    Get the WGS84 point / in the coordinate reference system
    epsg 4326 at in given a cartesian form of input i.e.
    providing the position of the destination point in relative
    meters east and meters north from the source point.
    If the source point is (0, 0) and you would like the coordinates
    of a point that lies 5 meters north and 3 meters west of the source,
    the bearing in degrees is hard to find on the fly. This function
    allows the input as follows:
    >>> get_cart_dest_point(source_point, -3, 5) # west is negative east
    Parameters
    ----------
    source_point: :shapely:`Shapely Point object<points>`
        The start point in WGS84 or epsg 4326 coordinates
    east_meters: :obj:`float`
        Meters to the east of source, negative number means west
    north_meters: :obj:`float`
        Meters to the north of source, negative number means south
    Returns
    -------
    :shapely:`Shapely Point object<points>`
        The point in WGS84 or epsg 4326 coordinates
        at the destination which is north_meters north
        of the source and east_meters east of source.
    """
    x_dist = abs(east_meters)
    y_dist = abs(north_meters)
    x_dir = (-90 if east_meters < 0
             else 90)
    y_dir = (180 if north_meters < 0
             else 0)
    intermediate_dest = get_dest_point(source_point, x_dist, x_dir)
    return get_dest_point(intermediate_dest, y_dist, y_dir)


def create_poly_from_source(source_point, left_m, right_m, up_m, down_m):
    """
    Create a rectangular polygon given a source point and the number of meters
    away from the source point the edges have to be.
    Parameters
    ----------
    source_point: :shapely:`Shapely Point object<points>`
        The start point in WGS84 or epsg 4326 coordinates
    left_m: :obj:`float`
        The distance from the source at which the left edge should be.
    right_m: :obj:`float`
        The distance from the source at which the right edge should be.
    up_m: :obj:`float`
        The distance from the source at which the upper edge should be.
    down_m: :obj:`float`
        The distance from the source at which the lower edge should be.
    Returns
    -------
    """
    poly_points = [get_cart_dest_point(source_point, -1*left_m, -1*down_m),
                   get_cart_dest_point(source_point, -1*left_m, up_m),
                   get_cart_dest_point(source_point, right_m, up_m),
                   get_cart_dest_point(source_point, right_m, -1*down_m),
                   get_cart_dest_point(source_point, -1*left_m, -1*down_m)]
    return Polygon(sum(map(list, (p.coords for p in poly_points)), []))
