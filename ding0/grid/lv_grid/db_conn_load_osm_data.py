"""
Load Buildings_with_Amenities, Building_wo_Amenity, Amenities_ni_Buildings, Way
From local DB
"""


from ding0.config.classes_db_conn.osm_load_classes import Buildings_with_Amenities, Building_wo_Amenity, Amenities_ni_Buildings, Way

from ding0.config.config_lv_grids_osm import get_config_osm

from sqlalchemy import func

import pandas as pd



def get_osm_ways(geo_area, session_osm):
    
    """ load ways from db for given polygon as geo_area_wkt """
    
    return session_osm.query(Way).filter(func.st_intersects(func.ST_GeomFromText(geo_area, get_config_osm('srid')), Way.geometry)) 


    


def get_osm_buildings_w_a(geo_area, session_osm):
    
    """ load buildings_with_amenities from db for given polygon as geo_area_wkt """
    
    buildings_w_a =  session_osm.query(Buildings_with_Amenities).filter(func.st_intersects(
        func.ST_GeomFromText(geo_area, get_config_osm('srid')), Buildings_with_Amenities.geo_center))
    
    buildings_w_a_sql_df = pd.read_sql(
        buildings_w_a.statement,
        con=session_osm.bind 
    )
    
    return buildings_w_a_sql_df



def get_osm_buildings_wo_a(geo_area, session_osm):
    
    """ load buildings_without_amenities from db for given polygon as geo_area_wkt """
    
    buildings_wo_a = session_osm.query(Building_wo_Amenity).filter(func.st_intersects(
        func.ST_GeomFromText(geo_area, get_config_osm('srid')), Building_wo_Amenity.geo_center))
    
    buildings_wo_a_sql_df = pd.read_sql(
        buildings_wo_a.statement,
        con=session_osm.bind 
    )
    
    return buildings_wo_a_sql_df



def get_osm_amenities_ni_Buildings(geo_area, session_osm):
    
    """ load amenities_notin_Buildings from db for given polygon as geo_area_wkt """
    
    amenities_ni_Buildings = session_osm.query(Amenities_ni_Buildings).filter(func.st_intersects(
        func.ST_GeomFromText(geo_area, get_config_osm('srid')), Amenities_ni_Buildings.geometry)) 
    
    amenities_ni_Buildings_sql_df = pd.read_sql(
        amenities_ni_Buildings.statement,
        con=session_osm.bind 
        # both ways are working. select the easier/ more appropriate one
        #con=engine_osm
    )
    
    return amenities_ni_Buildings_sql_df
