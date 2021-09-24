from shapely.geometry import MultiPoint, Point, shape
import pyproj
import pandas as pd



def convertCoords(coordx, coordy, from_proj, to_proj):
    
    """pyproj.transform from_proj, to_proj"""
    
    from_proj = pyproj.Proj(init='epsg:'+str(from_proj))
    to_proj   = pyproj.Proj(init='epsg:'+str(to_proj))
    
    x2,y2 = pyproj.transform(from_proj, to_proj, coordx, coordy)
    return pd.Series([x2, y2])




def get_Point_from_x_y(x, y):
    
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
    
    TODO: add points from generators?
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





def get_convex_hull_from_points(points):
    """ return convex hull for given points"""
    
    mpt = MultiPoint([shape(point) for point in points])
    
    return mpt.convex_hull
