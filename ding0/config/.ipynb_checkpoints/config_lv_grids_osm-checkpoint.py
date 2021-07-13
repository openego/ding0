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
        'avg_cluster_capacity' : 500,
        'additional_trafo_capacity' : 1.5
        
    }
        
    return config_osm[key]



def get_charging_station_columns():
    
    """ get columns for charging_stations """
    
    return ["socket:type2:output", "charging_station:output", "socket:chademo:output", 
            "socket:type2_cable:output", "socket:type2_combo:output"]



def get_peak_loads():
    
    """ get dictionary containing categories and peak loads """
    
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
        'charging_station' : 22.0,
        'residential' :   17.0847,
        'leftover'         :  0.1

    }
        
    return load_profiles
    

    
def get_load_profile_categories():
    
    """ get dictionary containing categorie types assigned to categories """
    
    load_profile_categories = {
        
        'lodging'      : ["lodging"],
    
        'education'    : ["education",
                           "language_school",
                           "kindergarten",
                           "school",
                           "art_school",
                           "university",
                           "college",
                           "driving_school",
                           "music_school"],

        'office'       : ["office",
                           "money_transfer",
                           "bank",
                           "prep_school",
                           "internet_cafe",
                           "courthouse",
                           "post_depot",
                           "post_office",
                           "studio",
                           "reception_desk",
                           "coworking_space",
                           "conference_centre",
                           "exhibition_centre",
                           "customs",
                           "supermarket",
                           "car_rental",
                           "car_wash",
                           "commercial",
                           "warehouse",
                           "police"],

        'retail'       : ["retail"],

        'fire_brigade' : ["fire_station"],

        'research'     : ["research"],

        'health_care'  : ["health_care",
                          "hospital",
                          "clinic",
                          "dentist",
                          "doctors",
                          "hospital",
                          "pharmacy",
                          "social_facility",
                          "veterinary"],

        'industry'     : ["industrial", 
                          "manufacture", 
                          "industry"],

        'hotel'        : ["hotel"],

        'aggriculture' : ["aggriculture",
                          "agricultural",
                          "barn",
                          "conservatory",
                          "cowshed",
                          "farm_auxiliary",
                          "greenhouse",
                          "stable"],

        'restaurant'   : ["restaurant",
                          "bar",
                          "biergarten",
                          "cafe",
                          "fast_food",
                          "ice_cream",
                          "pub",
                          "canteen"],

        'swimming'     : ["swimming_bath", 
                          "swimming_pool", 
                          "public_bath"],

        'sport'        : ["sport_building",
                          "sports_centre",
                          "grandstand",
                          "pavilion",
                          "riding_hall",
                          "sports_hall",
                          "stadium"],

        'technical'    : ["technique_building",
                          "digester",
                          "service",
                          "transformer_tower",
                          "water_tower",
                          "vehicle_inspection"],

        'residential'  : ["residential",
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
                          "yes"],

        'charging_station' : ["charging_station"],

        'public'       : ["public",
                          "public_building",
                          "arts_centre",
                          "civic",
                          "bakehouse",
                          "community_centre",
                          "library",
                          "toy_library",
                          "social_centre",
                          "parking",
                          "townhall",
                          "government",
                          "train_station",
                          "transportation",
                          "theatre",
                          "arts_centre",
                          "casino",
                          "cinema",
                          "nightclub",
                          "prison",
                          "cathedral",
                          "church",
                          "monastery",
                          "mosque",
                          "religious",
                          "synagogue",
                          "temple",
                          "place_of_worship",
                          "fuel"],
        
        
        'leftover'      : ["chapel, toilets"],

    }
    
    return load_profile_categories


