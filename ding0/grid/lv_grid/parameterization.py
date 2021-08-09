import pandas as pd

from geoalchemy2.shape import to_shape

from config.config_lv_grids_osm import get_peak_loads, get_load_profile_categories, get_config_osm 


# TODO: PARSE COLUMNS FOR charging_stations
#       THEY MAY PROVIDE A PEAK LOAD, e.g. 22 kW
#charging_station_columns = get_charging_station_columns()


def get_peak_load(category, load_profile_categories, load_profiles):
    
    """ Get peak load by given category of load profiles."""
    
    return load_profiles[load_profile_categories[category]]



def parameterize_by_load_profiles(amenities_ni_Buildings_sql_df, buildings_w_a_sql_df, buildings_wo_a_sql_df): 
    
    """Parameterize:
        buildings_w_a:  buildings_with_amenities
        buildings_wo_a: buildings_without_amenities
        amenities_ni_Buildings: amenities_not_in:buildings
        
       return:
         df_buildings_w_loads
         
         
        To distinguish between amenities and buildings, 
        due to they have different geometry (Point/ Polygon)
        amenities will init with p.x to be able to filter 
        where p.x == 'p.x' to set x,y in right way
    """
    
    # prepare parameterization info
    load_profile_categories = get_load_profile_categories()
    load_profiles = get_peak_loads()
    
    avg_load_res = 1.708470 # todo: set in config. will be updated when feeder are known.
    
    # set avg square meter from config
    avg_mxm = get_config_osm('avg_square_meters')
    
    
    # prepare pd.read_sql to concat as one df
    # preprocess amenities_ni_Buildings_sql_df for df
    if len(amenities_ni_Buildings_sql_df) > 0:
        amenities_ni_Buildings_sql_df.index = amenities_ni_Buildings_sql_df['osm_id']
        del amenities_ni_Buildings_sql_df['osm_id']
        del amenities_ni_Buildings_sql_df['tags']
        amenities_ni_Buildings_sql_df = amenities_ni_Buildings_sql_df.rename(
            {'amenity': 'category'}, axis=1)
        amenities_ni_Buildings_sql_df['area'] = avg_mxm
        amenities_ni_Buildings_sql_df['capacity'] = amenities_ni_Buildings_sql_df.apply(
            lambda row: get_peak_load(row.category, load_profile_categories, load_profiles) 
            * avg_mxm, axis=1)
        amenities_ni_Buildings_sql_df['x'], amenities_ni_Buildings_sql_df['y'] = 'p.x', 'p.y'
        amenities_ni_Buildings_sql_df['number_households'] = 0
        amenities_ni_Buildings_sql_df['raccordement_building'] = amenities_ni_Buildings_sql_df['geometry']
        concat_a = True
    else: concat_a=False
    
    # preprocess buildings_wo_a_sql_df for df
    if len(buildings_wo_a_sql_df) > 0:
        buildings_wo_a_sql_df['n_amenities_inside'] = 1
        buildings_wo_a_sql_df.index = buildings_wo_a_sql_df['osm_id']
        del buildings_wo_a_sql_df['osm_id']
        del buildings_wo_a_sql_df['tags']
        concat_b1 = True
    else: concat_b1=False
    
    # preprocess buildings_w_a_sql_df for df
    if len(buildings_w_a_sql_df) > 0:
        buildings_w_a_sql_df.index = buildings_w_a_sql_df['osm_id_amenity']
        buildings_w_a_sql_df = buildings_w_a_sql_df.rename({'geometry_building': 'geometry'}, axis=1)
        del buildings_w_a_sql_df['osm_id_amenity']
        del buildings_w_a_sql_df['building_tags']
        del buildings_w_a_sql_df['amenity_tags']
        concat_b2 = True
    else: concat_b2=False
    
    # concat buildings_wo_a_sql_df and buildings_w_a_sql_df
    if concat_b1 & concat_b2:
        buildings_w_a = pd.concat([buildings_wo_a_sql_df, buildings_w_a_sql_df]) 
        buildings_w_a_existing = True
    elif concat_b1:
        buildings_w_a = buildings_wo_a_sql_df.copy()
        buildings_w_a_existing = True
    elif concat_b2:
        buildings_w_a = buildings_w_a_sql_df.copy()
        buildings_w_a_existing = True
    else: buildings_w_a_existing=False
    
    
    existing_loads=True
    if buildings_w_a_existing:
        # preprocess concatted buildings_wo_a_sql_df and buildings_w_a_sql_df
        buildings_w_a['n_apartments'].replace(0,1, inplace=True)
        buildings_w_a['x'] = buildings_w_a['geo_center']
        buildings_w_a['y'] = 'p.y'
        buildings_w_a = buildings_w_a.rename({'building': 'category', 'n_apartments': 'number_households', 
                                              'geo_center': 'raccordement_building'}, axis=1)


        # set capacity for residentials. will be updated when connected feeders are known.
        # ensure has at least one household if residential
        buildings_w_a.loc[buildings_w_a.category.isin(load_profile_categories['residentials_list']), 
                          'number_households'].replace(0,1, inplace=True)
        buildings_w_a.loc[buildings_w_a.category.isin(load_profile_categories['residentials_list']), 
                          'capacity'] = buildings_w_a.number_households * avg_load_res

        # set capacity for ~residential
        buildings_w_a.loc[~buildings_w_a.category.isin(load_profile_categories['residentials_list']), 
                          'capacity'] = buildings_w_a.apply(lambda row: 
                                                            get_peak_load(row.category, 
                                                                          load_profile_categories, 
                                                                          load_profiles) 
                                                            * row.area * 1e-3 / row.n_amenities_inside, 
                                                            axis=1)
        if concat_a:
            buildings_w_loads_df = pd.concat([buildings_w_a, amenities_ni_Buildings_sql_df])
        else:
            buildings_w_loads_df = buildings_w_a.copy()
    
    elif concat_a:
        buildings_w_loads_df = amenities_ni_Buildings_sql_df.copy()
    else:
        existing_loads = False
    
    if existing_loads:
        
        # update to_shape(geometry), to_shape(raccordement_building)
        buildings_w_loads_df['geometry'] = buildings_w_loads_df.apply(
            lambda amenity: to_shape(amenity.geometry), axis=1)
        buildings_w_loads_df['raccordement_building'] = buildings_w_loads_df.apply(
            lambda building: to_shape(building.raccordement_building), axis=1)


        # for amenities update x and y
        buildings_w_loads_df.loc[buildings_w_loads_df.x == 'p.x', 'y'] = \
        buildings_w_loads_df.loc[buildings_w_loads_df.x == 'p.x'].apply(
            lambda amenity: amenity.geometry.y, axis=1)
        buildings_w_loads_df.loc[buildings_w_loads_df.x == 'p.x', 'x'] = \
        buildings_w_loads_df.loc[buildings_w_loads_df.x == 'p.x'].apply(
            lambda amenity: amenity.geometry.x, axis=1)

        # for amenities update x and y
        buildings_w_loads_df.loc[buildings_w_loads_df.y == 'p.y', 'x'] = \
        buildings_w_loads_df.loc[buildings_w_loads_df.y == 'p.y'].apply(
            lambda amenity: amenity.raccordement_building.x, axis=1)
        buildings_w_loads_df.loc[buildings_w_loads_df.y == 'p.y', 'y'] = \
        buildings_w_loads_df.loc[buildings_w_loads_df.y == 'p.y'].apply(
            lambda amenity: amenity.raccordement_building.y, axis=1)
    
        return buildings_w_loads_df
    
    else:
        return None
        
        
def parameterize_by_load_profiles_IT_DEPRECATED(buildings_w_a, buildings_wo_a, amenities_ni_Buildings): 
    
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
