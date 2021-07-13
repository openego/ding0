import pandas as pd

from geoalchemy2.shape import to_shape

from config.config_lv_grids_osm import get_peak_loads, get_load_profile_categories


# TODO: PARSE COLUMNS FOR charging_stations
#       THEY MAY PROVIDE A PEAK LOAD, e.g. 22 kW
#charging_station_columns = get_charging_station_columns()


def get_peak_load(category):
    
    """ Get peak load by given category of load profiles."""
    
    load_profile_categories  = get_load_profile_categories()
    load_profiles = get_peak_loads()

    for categorie_type, categories in load_profile_categories.items():

        if category in categories:

            return load_profiles[categorie_type]
        
        
        
def parameterize_by_load_profiles(buildings_w_a, buildings_wo_a, amenities_ni_Buildings): 
    
    """Parameterize:
        buildings_w_a:  buildings_with_amenities
        buildings_wo_a: buildings_without_amenities
        amenities_ni_Buildings: amenities_not_in:buildings
        
       return:
         df_buildings_w_loads
    """
    
    # TODO: CHECK IF TEMPORARY df needed    
    # assign nearest nodes more efficient thru dataframe instead calling iterative for each building
    # temporary: init empty df for buildings w. loads
    # will be replaced by oop when ding0 contains building-data
    # sqlalchemy does not allow: for building in buildings: building.capacity = new_Value 

    df_buildings_w_loads = pd.DataFrame(columns=['osm_id', 'category', 'capacity', 'number_households', 'x', 'y'])
    df_buildings_w_loads.index = df_buildings_w_loads['osm_id']
    del df_buildings_w_loads['osm_id']



    # TODO: set average 100m^2 as default size for amenities witouht known area
    # average square meter for buildings without area
    avg_mxm = 100
    
    
    load_profile_categories  = get_load_profile_categories()
    
    

    # check capacity
    for amenity in amenities_ni_Buildings:    

        p = to_shape(amenity.geometry)

        peak_load = get_peak_load(amenity.amenity)

        # if type does not exist in load profiles, set capacity=0 
        if peak_load is None:

            peak_load = 0



        if amenity.amenity == 'charging_station':

            df_buildings_w_loads.loc[amenity.osm_id] = [amenity.amenity, peak_load, 0, p.x, p.y]


        else:

            df_buildings_w_loads.loc[amenity.osm_id] = [amenity.amenity, peak_load * avg_mxm, 0, p.x, p.y]






    for building in buildings_wo_a:  


            p = to_shape(building.geometry)

            peak_load = get_peak_load(building.building)

            # if type does not exist in load profiles, set capacity=0 
            if peak_load is None:

                peak_load = 0


            number_households = building.n_apartments 


            # check if building is residential and add 1 household if no other number of households is known.
            # set peak load by load profile. will be updated when number of all households per feeder are known.
            if building.building in load_profile_categories['residential']:
                
                if building.building == 'yes':
                
                    building.building = 'residential'

                if number_households == 0:

                    number_households += 1

                df_buildings_w_loads.loc[building.osm_id] = [building.building, peak_load * avg_mxm * number_households * 1e-3, number_households, p.x, p.y]



            # parameterize all categories but residentials
            else:

                # peak_load = peak_load_per_square_meter * square_meter * 1e-3.        
                df_buildings_w_loads.loc[building.osm_id] = [building.building, peak_load * building.area * 1e-3, number_households, p.x, p.y]




    for building in buildings_w_a: 


            p = to_shape(building.geometry)

            peak_load = get_peak_load(building.building)

            # if type does not exist in load profiles, set capacity=0 
            if peak_load is None:

                peak_load = 0


            number_households = building.n_apartments


            # check if building is residential and add 1 household if no other number of households is known.
            # set peak load 0. will be updated when number of all households per feeder are known.
            if building.building in load_profile_categories['residential']:
                
                if building.building == 'yes':
                
                    building.building = 'residential'

                if number_households == 0:

                    number_households += 1

                df_buildings_w_loads.loc[building.osm_id] = [building.building, peak_load * avg_mxm * number_households * 1e-3, number_households, p.x, p.y]



            # parameterize all categories but residentials
            else:

                # peak_load in kW. if building contains multiple amenities, area is shared unifromly.
                # peak_load = peak_load_per_square_meter * square_meter / n_amenities_inside * 1e-3.        

                peak_load = get_peak_load(building.building)

                # if type does not exist in load profiles, set capacity=0 
                if peak_load is None:

                    peak_load = 0

                df_buildings_w_loads.loc[building.osm_id] = [building.building, peak_load * building.area / building.n_amenities_inside * 1e-3, number_households, p.x, p.y]


                
                
    return df_buildings_w_loads
