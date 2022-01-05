import pandas as pd

from geoalchemy2.shape import to_shape

from config.config_lv_grids_osm import get_peak_loads, get_load_profile_categories, get_config_osm 



def get_peak_load(category, load_profile_categories, load_profiles, area, n_amenities_inside, number_households):
    
    """ 
    Get peak load by given category of load profiles.
    Multiply by 1e-3 to transform to kW.
    """
    
    if category in load_profile_categories['residentials_list']:
    
        # capacity for residentials will be updated after clustering
        # due to number of residentials per feeder are mandatory
        return load_profiles[load_profile_categories[category]] * 100 * number_households * 1e-3
    
    if category not in load_profile_categories:
        
        print(category, 'is not matched in config and will be categorized as leftover')
        category = 'leftover'
        
    return load_profiles[load_profile_categories[category]] * area / n_amenities_inside * 1e-3


def get_peak_load_diversity(buildings):
    """ get get_peak_load_diversity for each given area """
    load_profile_categories = get_load_profile_categories()
    diversity_factor = get_config_osm('diversity_factor_not_residential')

    total_number_hh = buildings.loc[buildings['category'].isin(load_profile_categories['residentials_list'])]['number_households'].sum()
    peak_load_res = get_peak_load_for_residential(total_number_hh) * total_number_hh
    peak_load_not_res = buildings.loc[~buildings['category'].isin(load_profile_categories['residentials_list']), 'capacity'].sum() * diversity_factor
    
    return peak_load_res + peak_load_not_res


def get_peak_load_for_residential(number_residentials):
    '''
    calculate peak load in MW for residential depending on its number at feeder

    SOURCE: Dissertation
    AUTHOR: Thomas Stetz
    TITEL: Autonomous Voltage Control Strategies in Distribution Grids with Photovoltaic Systems
    - Technical and Economic Assessment - Stetz
    Page: 14 
    Formula: (2.4)

    $\sqrt[k]{n}$ = k root n or check mardown
    pl = a - ( b / $\sqrt[k]{n}$)

    e.g. calculation        tranformation kW to MW before return
    number residentials     peak load [kW]
                     10     2.5291793904118034
    Threshold after  35     1.7180038451156543
    Threshold exceed 36     1.706869146916332
                     70     1.500011219348335
                    110     1.406355083981595
    
    return peak_load_per_residential
    '''
    a = 1.16221
    b = - 7.14717
    k = 1.39203

    pl = a - b / number_residentials**(1/k)

    return pl


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

    # set avg square meter from config
    avg_mxm = get_config_osm('avg_square_meters')

    # prepare pd.read_sql to concat as one df
    # preprocess amenities_ni_Buildings_sql_df for df
    if len(amenities_ni_Buildings_sql_df):
        amenities_ni_Buildings_sql_df.index = amenities_ni_Buildings_sql_df['osm_id']
        # del amenities_ni_Buildings_sql_df['osm_id']
        amenities_ni_Buildings_sql_df = \
            amenities_ni_Buildings_sql_df.rename({'osm_id': 'osm_id_amenity'}, axis=1)
        amenities_ni_Buildings_sql_df['number_households'] = 1
        amenities_ni_Buildings_sql_df = amenities_ni_Buildings_sql_df.rename(
            {'amenity': 'category'}, axis=1)
        amenities_ni_Buildings_sql_df['area'] = avg_mxm
        amenities_ni_Buildings_sql_df['n_amenities_inside'] = 1
        amenities_ni_Buildings_sql_df['capacity'] = amenities_ni_Buildings_sql_df.apply(
            lambda row: get_peak_load(row.category, load_profile_categories, 
                                      load_profiles, row.area, row.n_amenities_inside,
                                      row.number_households), axis=1)
        amenities_ni_Buildings_sql_df['x'], amenities_ni_Buildings_sql_df['y'] = 'p.x', 'p.y'
        amenities_ni_Buildings_sql_df['raccordement_building'] = amenities_ni_Buildings_sql_df['geometry']
        concat_a = True
    else: concat_a=False

    # preprocess buildings_wo_a_sql_df for df
    if len(buildings_wo_a_sql_df):
        buildings_wo_a_sql_df['n_amenities_inside'] = 1
        buildings_wo_a_sql_df.index = buildings_wo_a_sql_df['osm_id']
        # del buildings_wo_a_sql_df['osm_id']
        buildings_wo_a_sql_df = buildings_wo_a_sql_df.rename({'osm_id': 'osm_id_building'}, axis=1)
        concat_b1 = True
    else: concat_b1=False

    # preprocess buildings_w_a_sql_df for df
    if len(buildings_w_a_sql_df):
        buildings_w_a_sql_df.index = buildings_w_a_sql_df['osm_id_amenity']
        buildings_w_a_sql_df = buildings_w_a_sql_df.rename({'osm_id': 'osm_id_building'}, axis=1)
        buildings_w_a_sql_df = buildings_w_a_sql_df.rename({'geometry_building': 'geometry'}, axis=1)
        # buildings_w_a_sql_df['osm_id_amenity']
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
        
        buildings_w_a['capacity'] = buildings_w_a.apply(lambda row: 
                                                        get_peak_load(row.category, 
                                                                      load_profile_categories, 
                                                                      load_profiles,
                                                                      row.area,
                                                                      row.n_amenities_inside,
                                                                      row.number_households),
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
        
        # check for duplicates
        ids = buildings_w_loads_df.index
        dupl_ids = list(set(buildings_w_loads_df[ids.isin(ids[ids.duplicated()])].index.tolist()))

        if len(dupl_ids) > 0: # drop dupl if exists e.g. one amenity is located in multiple buildings.

            dupl_df = buildings_w_loads_df.loc[buildings_w_loads_df.index.isin(dupl_ids)]

            for ix, dupl_id in enumerate(dupl_ids):

                if ix < 1:

                    # todo dff rename but not dupl_df = dupl_df[] du to reameining with 1 entry
                    dupl_dff = dupl_df[(dupl_df.index==dupl_id) & (dupl_df.area!=dupl_df.loc[dupl_df.index==dupl_id].area.max())]

                else:

                    tempf_df = dupl_df[(dupl_df.index==dupl_id) & (dupl_df.area!=dupl_df.loc[dupl_df.index==dupl_id].area.max())]
                    dupl_dff = pd.concat([dupl_dff, tempf_df])
                    
            
            buildings_w_loads_df = buildings_w_loads_df[~buildings_w_loads_df.index.isin(dupl_ids)]

            buildings_w_loads_df = pd.concat([buildings_w_loads_df, dupl_dff])

        
        # update to_shape(geometry), to_shape(raccordement_building)
        buildings_w_loads_df['geometry'] = buildings_w_loads_df.apply(
            lambda amenity: to_shape(amenity.geometry), axis=1)
        buildings_w_loads_df['raccordement_building'] = buildings_w_loads_df.apply(
            lambda building: to_shape(building.raccordement_building), axis=1)
        # to_shape only working if not nan
        if 'geometry_amenity' in buildings_w_loads_df.columns:
            buildings_w_loads_df.loc[buildings_w_loads_df.geometry_amenity.notnull(), 'geometry_amenity'] = \
            buildings_w_loads_df.loc[buildings_w_loads_df.geometry_amenity.notnull()].apply(
                lambda building: to_shape(building.geometry_amenity), axis=1)


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
        
        # replace yes to residentials
        buildings_w_loads_df['category'].replace('yes','residential', inplace=True)
    
        return buildings_w_loads_df
    
    else:
        return None
