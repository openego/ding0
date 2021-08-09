"""
config file for synthetic low voltage grids 
parameterized by load profiles.

"""

def get_config_osm(key):
    
    """
    return config for osm processing.
    ...
    ...
    mv_lv_threshold_capacity: loads with capacity < 200 kW are connected in grid
                              loads wi. capacity >= 200 kW are separately
    
    avg_cluster_capacity    : n_cluster = sum(peak_load) / avg_cluster_capacity
                              Clusters are NOT homogenous
                              
    additional_trafo_capacity: 1.5; so, transformers have max load factor of 70%.
                               capacity_to_cover = capacity_of_area * 1.5
                               
                              
    """

    config_osm = {
        
        'srid' : 4326,
        'EARTH_RADIUS_M' : 6_371_009,
        'mv_lv_threshold_capacity' : 200,
        'additional_trafo_capacity' : 1.5,
        'avg_trafo_size' : 500,
        'avg_square_meters' : 100,
        
    }
        
    return config_osm[key]



def get_charging_station_columns():
    
    """ get columns for charging_stations """
    
    return ["socket:type2:output", "charging_station:output", "socket:chademo:output", 
            "socket:type2_cable:output", "socket:type2_combo:output"]



def get_peak_loads():
    
    """ 
    get dictionary containing categories and peak loads 
    due to each amenity as average square meters of 100
    but for charging_stations we use 22 kW, divide by
    average square meters.
    """
    
    
    
    load_profiles = {

        'lodging'      :  73.0,
        'education'    :  62.2,
        'office'       :  64.9,
        'retail'       : 128.9,
        'fire_brigade' :  12.93,
        'research'     :  91.51,
        'health_care'  : 109.3,
        'industry'     : 164.5,
        'hotel'        :  62.7,
        'aggriculture' :  38.29,
        'restaurant'   :  91.51,
        'swimming'     :  88.50,
        'sport'        :  72.72,
        'technical'    :  13.25,
        'public'       :  115.8,
        'charging_station' : 22.0 / get_config_osm('avg_square_meters'), 
        'residential' :   17.0847,
        'leftover'         :  0.1

    }
        
    return load_profiles
    



def get_load_profile_categories():

    """ get dictionary containing categorie types assigned to categories """

    load_profile_categories = {
        "lodging": "lodging",
        "education": "education",
        "language_school": "education",
        "kindergarten": "education",
        "school": "education",
        "art_school": "education",
        "university": "education",
        "college": "education",
        "driving_school": "education",
        "music_school": "education",
        "office": "office",
        "money_transfer": "office",
        "bank": "office",
        "prep_school": "office",
        "internet_cafe": "office",
        "courthouse": "office",
        "post_depot": "office",
        "post_office": "office",
        "studio": "office",
        "reception_desk": "office",
        "coworking_space": "office",
        "conference_centre": "office",
        "exhibition_centre": "office",
        "customs": "office",
        "supermarket": "office",
        "car_rental": "office",
        "car_wash": "office",
        "commercial": "office",
        "warehouse": "office",
        "police": "office",
        "retail": "retail",
        "fire_station": "fire_brigade",
        "research": "research",
        "health_care": "health_care",
        "hospital": "health_care",
        "clinic": "health_care",
        "dentist": "health_care",
        "doctors": "health_care",
        "hospital": "health_care",
        "pharmacy": "health_care",
        "social_facility": "health_care",
        "veterinary": "health_care",
        "industrial": "industry",
        "manufacture": "industry",
        "industry": "industry",
        "hotel": "hotel",
        "aggriculture": "aggriculture",
        "agricultural": "aggriculture",
        "barn": "aggriculture",
        "conservatory": "aggriculture",
        "cowshed": "aggriculture",
        "farm_auxiliary": "aggriculture",
        "greenhouse": "aggriculture",
        "stable": "aggriculture",
        "restaurant": "restaurant",
        "bar": "restaurant",
        "biergarten": "restaurant",
        "cafe": "restaurant",
        "fast_food": "restaurant",
        "ice_cream": "restaurant",
        "pub": "restaurant",
        "canteen": "restaurant",
        "swimming_bath": "swimming",
        "swimming_pool": "swimming",
        "public_bath": "swimming",
        "sport_building": "sport",
        "sports_centre": "sport",
        "grandstand": "sport",
        "pavilion": "sport",
        "riding_hall": "sport",
        "sports_hall": "sport",
        "stadium": "sport",
        "technique_building": "technical",
        "digester": "technical",
        "service": "technical",
        "transformer_tower": "technical",
        "water_tower": "technical",
        "vehicle_inspection": "technical",
        "residential": "residential",
        "retirement_home": "residential",
        "apartments": "residential",
        "house": "residential",
        "brothel": "residential",
        "bungalow": "residential",
        "detached": "residential",
        "semidetached_house": "residential",
        "semi": "residential",
        "dormitory": "residential",
        "farm": "residential",
        "semidetached_house": "residential",
        "terrace": "residential",
        "yes": "residential",
        "charging_station": "charging_station",
        "public": "public",
        "public_building": "public",
        "arts_centre": "public",
        "civic": "public",
        "bakehouse": "public",
        "community_centre": "public",
        "library": "public",
        "toy_library": "public",
        "social_centre": "public",
        "parking": "public",
        "townhall": "public",
        "government": "public",
        "train_station": "public",
        "transportation": "public",
        "theatre": "public",
        "arts_centre": "public",
        "casino": "public",
        "cinema": "public",
        "nightclub": "public",
        "prison": "public",
        "cathedral": "public",
        "church": "public",
        "monastery": "public",
        "mosque": "public",
        "religious": "public",
        "synagogue": "public",
        "temple": "public",
        "fuel": "public",
        "chapel": "leftover",
        "place_of_worship": "leftover",
        "toilets": "leftover",
        "residentials_list": [
            "residential",
            "retirement_home",
            "apartments",
            "house",
            "brothel",
            "bungalow",
            "detached",
            "semidetached_house",
            "semi",
            "dormitory",
            "farm",
            "semidetached_house",
            "terrace",
            "yes",
        ],
    } 
    
    return load_profile_categories
    
