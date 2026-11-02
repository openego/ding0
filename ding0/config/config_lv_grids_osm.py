"""
config file for synthetic low voltage grids 
parameterized by load profiles.

"""

def get_config_osm(key):
    """
    Returns a parameter of the OSM-based LV grid generation.

    The parameters are taken from the section `lv_grids_osm` of the active
    planning profile, see :mod:`ding0.tools.planning_profiles`.

    Parameters
    ----------
    key : :obj:`str`
        Parameter name

    Returns
    -------
    :obj:`int`, :obj:`float`, :obj:`list` or :obj:`str`
        Parameter value
    """
    from ding0.tools.planning_profiles import get_osm_parameter
    return get_osm_parameter(key)



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
        'charging_station' : 22000.0 / get_config_osm('avg_square_meters'), 
        'residential' :   17.0847,
        'leftover'         :  0.1

    }
        
    return load_profiles
    



def get_load_profile_categories():

    """ get dictionary containing categorie types assigned to categories """
    #TODO exclude leftover category from building import

    load_profile_categories = {
        "lodging": "lodging",
        "education": "education",
        "language_school": "education",
        "childcare": "education",
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
        "food_court": "restaurant",
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
        "terrace": "residential",
        "yes": "residential",
        "charging_station": "charging_station",
        "public": "public",
        "archive": "public",
        "public_building": "public",
        "events_venue": "public",
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
        "leftover": "leftover",
        "chapel": "leftover",
        "place_of_worship": "leftover",
        "toilets": "leftover",
        "recycling": "industry",
        "waste_transfer_station": "industry",
        "shelter": "leftover",
        "kiosk": "retail",
        "nursing_home": "health_care",
        "planetarium": "public",
        "gambling": "public",
        "static_caravan": "leftover",
        "bicycle_parking": "leftover",
        "parish_hall": "public",
        "embassy": "public",
        "emergency_service": "health_care",
        "bench": "leftover", #TODO: drop amenities with obviously no consumption
        "love_hotel": "public",
        "stripclub": "public",
        "swingerclub": "public",
        "research_institute": "research",
        "school;driving_school": "education",
        "sailing_school": "education",
        "job_centre": "public",
        "dive_centre": "swimming",
        "student_accommodation": "lodging",
        "animal_boarding": "leftover",
        "bicycle_repair_station": "leftover",
        "parking_space": "leftover", #TODO: drop amenities with obviously no consumption
        "motorcycle_parking": "leftover",  # TODO: drop amenities with obviously no consumption
        "dressing_room": "leftover",
        "funeral_hall": "leftover",
        "waste_disposal": "leftover", # TODO: drop amenities with obviously no consumption
        "station": "public",
        "gatehouse": "leftover",
        "fountain": "leftover",
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
    
