import pandas as pd

from geoalchemy2.shape import to_shape

from config.config_lv_grids_osm import get_peak_loads, get_load_profile_categories, get_config_osm 


# TODO: PARSE COLUMNS FOR charging_stations
#       THEY MAY PROVIDE A PEAK LOAD, e.g. 22 kW
#charging_station_columns = get_charging_station_columns()


def get_peak_load(category, load_profile_categories, load_profiles):
    
    """ Get peak load by given category of load profiles."""
    
    return load_profiles[load_profile_categories[category]]
        
        
        
def parameterize_by_load_profiles(buildings_w_a, buildings_wo_a, amenities_ni_Buildings): 
    
    """Parameterize:
        buildings_w_a:  buildings_with_amenities
        buildings_wo_a: buildings_without_amenities
        amenities_ni_Buildings: amenities_not_in:buildings
        
       return:
         df_buildings_w_loads
         
         
        To distinguish between amenities and buildings, due to they have different geometry (Point/ Polygon)
        amenities will init with p.x to be able to filter for where p.x == 'p.x' to set x,y in right way
    """
    
    # prepare parameterization info
    load_profile_categories = get_load_profile_categories()
    load_profiles = get_peak_loads()
    
    # set avg square meter from config
    avg_mxm = get_config_osm('avg_square_meters')
    
    # TODO: CHECK IF TEMPORARY df needed    
    # assign nearest nodes more efficient thru dataframe instead calling iterative for each building
    # temporary: init empty df for buildings w. loads
    # will be replaced by oop when ding0 contains building-data
    # sqlalchemy does not allow: for building in buildings: building.capacity = new_Value 

    df_buildings_w_loads = pd.DataFrame(columns=['osm_id', 'category', 'capacity', 'area', 'number_households', 
                                                 'x', 'y', 'geometry', 'raccordement_building'])
    df_buildings_w_loads.index = df_buildings_w_loads['osm_id']
    del df_buildings_w_loads['osm_id']
    
    

    # check capacity
    for amenity in amenities_ni_Buildings:    

        peak_load = get_peak_load(amenity.amenity, load_profile_categories, load_profiles)

        # if type does not exist in load profiles, set capacity=0 
        if peak_load is None:

            peak_load = 0


        df_buildings_w_loads.loc[amenity.osm_id] = [amenity.amenity, peak_load * avg_mxm, avg_mxm, 0, 
                                                        'p.x', 'p.y', amenity.geometry, amenity.geometry]




    for building in buildings_w_a:

        peak_load = get_peak_load(building.building, load_profile_categories, load_profiles)

        # if type does not exist in load profiles, set capacity=0 
        if peak_load is None:

            peak_load = 0


        number_households = building.n_apartments


        # check if building is residential and add 1 household if no other number of households is known.
        # set peak load 0. will be updated when number of all households per feeder are known.
        if building.building in load_profile_categories['residentials_list']:

            if building.building == 'yes':

                building.building = 'residential'

            if number_households == 0:

                number_households += 1

            df_buildings_w_loads.loc[building.osm_id_amenity] = [building.building, peak_load * avg_mxm * 
                                                                 number_households * 1e-3, avg_mxm, 
                                                                 number_households, building.geo_center, 'p.y', 
                                                                 building.geometry_building, building.geo_center]



        # parameterize all categories but residentials
        else:

            # peak_load in kW. if building contains multiple amenities, area is shared unifromly.
            # peak_load = peak_load_per_square_meter * square_meter / n_amenities_inside * 1e-3.

            df_buildings_w_loads.loc[building.osm_id_amenity] = [building.building, peak_load * building.area / 
                                                                 building.n_amenities_inside * 1e-3, building.area,
                                                                 number_households, building.geo_center, 'p.y', 
                                                                 building.geometry_building, building.geo_center]






    for building in buildings_wo_a:  

        peak_load = get_peak_load(building.building, load_profile_categories, load_profiles)

        # if type does not exist in load profiles, set capacity=0 
        if peak_load is None:

            peak_load = 0


        number_households = building.n_apartments 


        # check if building is residential and add 1 household if no other number of households is known.
        # set peak load by load profile. will be updated when number of all households per feeder are known.
        if building.building in load_profile_categories['residentials_list']:

            if building.building == 'yes':

                building.building = 'residential'

            if number_households == 0:

                number_households += 1

            df_buildings_w_loads.loc[building.osm_id] = [building.building, peak_load * avg_mxm * 
                                                         number_households * 1e-3, avg_mxm, 
                                                         number_households, building.geo_center, 'p.y', 
                                                         building.geometry, building.geo_center]



        # parameterize all categories but residentials
        else:

            # peak_load = peak_load_per_square_meter * square_meter * 1e-3.        
            df_buildings_w_loads.loc[building.osm_id] = [building.building, peak_load * building.area * 1e-3, 
                                                         building.area, number_households, building.geo_center, 
                                                         'p.y', building.geometry, building.geo_center]

    

    # update to_shape(geometry), to_shape(raccordement_building)
    df_buildings_w_loads['geometry'] = df_buildings_w_loads.apply(
        lambda amenity: to_shape(amenity.geometry), axis=1)
    df_buildings_w_loads['raccordement_building'] = df_buildings_w_loads.apply(
        lambda building: to_shape(building.raccordement_building), axis=1)


    # for amenities update x and y
    df_buildings_w_loads.loc[df_buildings_w_loads.x == 'p.x', 'y'] = \
    df_buildings_w_loads.loc[df_buildings_w_loads.x == 'p.x'].apply(
        lambda amenity: amenity.geometry.y, axis=1)
    df_buildings_w_loads.loc[df_buildings_w_loads.x == 'p.x', 'x'] = \
    df_buildings_w_loads.loc[df_buildings_w_loads.x == 'p.x'].apply(
        lambda amenity: amenity.geometry.x, axis=1)

    # for amenities update x and y
    df_buildings_w_loads.loc[df_buildings_w_loads.y == 'p.y', 'x'] = \
    df_buildings_w_loads.loc[df_buildings_w_loads.y == 'p.y'].apply(
        lambda amenity: amenity.raccordement_building.x, axis=1)
    df_buildings_w_loads.loc[df_buildings_w_loads.y == 'p.y', 'y'] = \
    df_buildings_w_loads.loc[df_buildings_w_loads.y == 'p.y'].apply(
        lambda amenity: amenity.raccordement_building.y, axis=1)

    
    return df_buildings_w_loads
