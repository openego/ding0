from shapely.geometry import MultiPoint, Point, shape



def get_Point_from_x_y(x, y):
    
    """ 
    get point for given x,y
    """
    
    return Point(x,y)




def get_points_in_load_area(geometry):
    
    """ 
    get all points in a load area
    based on buildings, e.g.:
        buildings_with_amenities
        buildings_without_amenities
        amenities_not_in_buildings
    
    TODO: add points from generators?
    """

    point_lists = []
    for geo in geometry:

        if geo.geom_type == 'Polygon':
            point_lists += [Point(point) for point in geo.exterior.coords[:-1]]

        elif geo.geom_type == 'Point':
            point_lists.append(geo)

        else:

            raise IOError('Shape is not a polygon neither a point.')
    
    
    return point_lists





def get_convex_hull_from_points(points):
    """ return convex hull for given points"""
    
    mpt = MultiPoint([shape(point) for point in points])
    
    return mpt.convex_hull



