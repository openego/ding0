"""
Load Buildings_with_Amenities, Building_wo_Amenity, Amenities_ni_Buildings, Way
From local DB
"""


from config.classes_db_conn.osm_load_classes import Buildings_with_Amenities, Building_wo_Amenity, Amenities_ni_Buildings, Way

from config.config_lv_grids_osm import get_config_osm

from sqlalchemy import func



def get_osm_ways(geo_area, session_osm):
    
    """ load ways from db for given polygon as geo_area_wkt """
    
    ways = session_osm.query(Way).filter(func.st_intersects(func.ST_GeomFromText(geo_area, get_config_osm('srid')), Way.geometry)) 
    
    
    return ways
