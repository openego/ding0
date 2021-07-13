from geoalchemy2.shape import to_shape
from shapely.geometry import MultiPoint, Point, shape


def get_points_in_load_area(buildings_df):
    
    """ 
    get all points in a load area
    based on buildings, e.g.:
        buildings_with_amenities
        buildings_without_amenities
        amenities_not_in_buildings
    
    TODO: add points from generators
    """

    points = []

    for i, row in buildings_df.iterrows():
        
        points.append(Point(row.x, row.y))
    
    
    return points





def get_convex_hull_from_points(points):
    """ return convex hull for given points"""
    
    mpt = MultiPoint([shape(point) for point in points])
    
    return mpt.convex_hull
