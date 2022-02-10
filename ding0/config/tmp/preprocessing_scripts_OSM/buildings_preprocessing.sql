

DROP TABLE if exists amenities_filtered


-- create table containing filtered amenities
CREATE TABLE amenities_filtered as
    select pop.osm_id, pop.amenity, pop."name", st_transform(ST_SetSRID(pop.way, 3857), 3035) as geometry, pop.tags
    FROM public.planet_osm_point pop
    where pop.amenity like 'bar'
    or pop.amenity like 'biergarten'
    or pop.amenity like 'cafe'
    or pop.amenity like 'fast_food'
    or pop.amenity like 'food_court'
    or pop.amenity like 'ice_cream'
    or pop.amenity like 'pub'
    or pop.amenity like 'restaurant'
    or pop.amenity like 'college'
    or pop.amenity like 'driving_school'
    or pop.amenity like 'kindergarten'
    or pop.amenity like 'language_school'
    or pop.amenity like 'library'
    or pop.amenity like 'toy_library'
    or pop.amenity like 'music_school'
    or pop.amenity like 'school'
    or pop.amenity like 'university'
    or pop.amenity like 'car_wash'
    or pop.amenity like 'vehicle_inspection'
    or pop.amenity like 'charging_station'
    or pop.amenity like 'fuel'
    or pop.amenity like 'bank'
    or pop.amenity like 'clinic'
    or pop.amenity like 'dentist'
    or pop.amenity like 'doctors'
    or pop.amenity like 'hospital'
    or pop.amenity like 'nursing_home'
    or pop.amenity like 'pharmacy'
    or pop.amenity like 'social_facility'
    or pop.amenity like 'veterinary'
    or pop.amenity like 'arts_centre'
    or pop.amenity like 'brothel'
    or pop.amenity like 'casino'
    or pop.amenity like 'cinema'
    or pop.amenity like 'community_centre'
    or pop.amenity like 'conference_centre'
    or pop.amenity like 'events_venue'
    or pop.amenity like 'gambling'
    or pop.amenity like 'love_hotel'
    or pop.amenity like 'nightclub'
    or pop.amenity like 'planetarium'
    or pop.amenity like 'social_centre'
    or pop.amenity like 'stripclub'
    or pop.amenity like 'studio'
    or pop.amenity like 'swingerclub'
    or pop.amenity like 'exhibition_centre'
    or pop.amenity like 'theatre'
    or pop.amenity like 'courthouse'
    or pop.amenity like 'embassy'
    or pop.amenity like 'fire_station'
    or pop.amenity like 'police'
    or pop.amenity like 'post_depot'
    or pop.amenity like 'post_office'
    or pop.amenity like 'prison'
    or pop.amenity like 'ranger_station'
    or pop.amenity like 'townhall'
    or pop.amenity like 'animal_boarding'
    or pop.amenity like 'childcare'
    or pop.amenity like 'dive_centre'
    or pop.amenity like 'funeral_hall'
    or pop.amenity like 'gym'
    or pop.amenity like 'internet_cafe'
    or pop.amenity like 'kitchen'
    or pop.amenity like 'monastery'
    or pop.amenity like 'place_of_mourning'
    or pop.amenity like 'place_of_worship'

-- create index
CREATE INDEX ON amenities_filtered USING gist (geometry);

-- ^- this table will be droppped at a later point due to the need is only for preprocessing steps



DROP TABLE if exists buildings_filtered


-- create table containing filtered buildings
CREATE TABLE buildings_filtered as
    select pop.osm_id, pop.amenity, pop.building, pop."name", st_transform(ST_SetSRID(pop.way, 3857), 3035) as geometry, pop.way_area as "area", pop.tags
    from public.planet_osm_polygon pop
    where pop.building like 'yes'
    or pop.building like 'apartments'
    or pop.building like 'detached'
    or pop.building like 'dormitory'
    or pop.building like 'farm'
    or pop.building like 'hotel'
    or pop.building like 'house'
    or pop.building like 'residential'
    or pop.building like 'semidetached_house'
    or pop.building like 'static_caravan'
    or pop.building like 'terrace'
    or pop.building like 'commercial'
    or pop.building like 'industrial'
    or pop.building like 'kiosk'
    or pop.building like 'office'
    or pop.building like 'retail'
    or pop.building like 'supermarket'
    or pop.building like 'warehouse'
    or pop.building like 'cathedral'
    or pop.building like 'chapel'
    or pop.building like 'church'
    or pop.building like 'monastery'
    or pop.building like 'mosque'
    or pop.building like 'presbytery'
    or pop.building like 'religious'
    or pop.building like 'shrine'
    or pop.building like 'synagogue'
    or pop.building like 'temple'
    or pop.building like 'bakehouse'
    or pop.building like 'civic'
    or pop.building like 'fire_station'
    or pop.building like 'government'
    or pop.building like 'hospital'
    or pop.building like 'public'
    or pop.building like 'train_station'
    or pop.building like 'transportation'
    or pop.building like 'kindergarten'
    or pop.building like 'school'
    or pop.building like 'university'
    or pop.building like 'college'
    or pop.building like 'barn'
    or pop.building like 'conservatory'
    or pop.building like 'farm_auxiliary'
    or pop.building like 'stable'
    or pop.building like 'pavilion'
    or pop.building like 'riding_hall'
    or pop.building like 'sports_hall'
    or pop.building like 'stadium'
    or pop.building like 'digester'
    or pop.building like 'service'
    or pop.building like 'transformer_tower'
    or pop.building like 'military'
    or pop.building like 'gatehouse'
    or pop.amenity like 'bar'
    or pop.amenity like 'biergarten'
    or pop.amenity like 'cafe'
    or pop.amenity like 'fast_food'
    or pop.amenity like 'food_court'
    or pop.amenity like 'ice_cream'
    or pop.amenity like 'pub'
    or pop.amenity like 'restaurant'
    or pop.amenity like 'college'
    or pop.amenity like 'driving_school'
    or pop.amenity like 'kindergarten'
    or pop.amenity like 'language_school'
    or pop.amenity like 'library'
    or pop.amenity like 'toy_library'
    or pop.amenity like 'music_school'
    or pop.amenity like 'school'
    or pop.amenity like 'university'
    or pop.amenity like 'car_wash'
    or pop.amenity like 'vehicle_inspection'
    or pop.amenity like 'charging_station'
    or pop.amenity like 'fuel'
    or pop.amenity like 'bank'
    or pop.amenity like 'clinic'
    or pop.amenity like 'dentist'
    or pop.amenity like 'doctors'
    or pop.amenity like 'hospital'
    or pop.amenity like 'nursing_home'
    or pop.amenity like 'pharmacy'
    or pop.amenity like 'social_facility'
    or pop.amenity like 'veterinary'
    or pop.amenity like 'arts_centre'
    or pop.amenity like 'brothel'
    or pop.amenity like 'casino'
    or pop.amenity like 'cinema'
    or pop.amenity like 'community_centre'
    or pop.amenity like 'conference_centre'
    or pop.amenity like 'events_venue'
    or pop.amenity like 'gambling'
    or pop.amenity like 'love_hotel'
    or pop.amenity like 'nightclub'
    or pop.amenity like 'planetarium'
    or pop.amenity like 'social_centre'
    or pop.amenity like 'stripclub'
    or pop.amenity like 'studio'
    or pop.amenity like 'swingerclub'
    or pop.amenity like 'exhibition_centre'
    or pop.amenity like 'theatre'
    or pop.amenity like 'courthouse'
    or pop.amenity like 'embassy'
    or pop.amenity like 'fire_station'
    or pop.amenity like 'police'
    or pop.amenity like 'post_depot'
    or pop.amenity like 'post_office'
    or pop.amenity like 'prison'
    or pop.amenity like 'ranger_station'
    or pop.amenity like 'townhall'
    or pop.amenity like 'animal_boarding'
    or pop.amenity like 'childcare'
    or pop.amenity like 'dive_centre'
    or pop.amenity like 'funeral_hall'
    or pop.amenity like 'gym'
    or pop.amenity like 'internet_cafe'
    or pop.amenity like 'kitchen'
    or pop.amenity like 'monastery'
    or pop.amenity like 'place_of_mourning'
    or pop.amenity like 'place_of_worship'


-- create index
CREATE INDEX ON buildings_filtered USING gist (geometry);

-- ^- this table will be droppped at a later point due to the need is only for preprocessing steps




-- merge n_apartments representing households to buildings_filtered
-- first a temp table is needed due to wanna add a count to share n_apartments for n_buildings
-- all buildings within a radius of ~77.7 meters


drop table if exists buildings_with_res_tmp

-- create temporary table of buildings containing zensus data (residentials)
CREATE TABLE buildings_with_res_tmp as
    select * from (
        select buildings.osm_id, buildings.amenity, buildings.building, buildings.name, buildings.geometry, buildings.area, buildings.tags, zensus_apartments.n_apartments, zensus_apartments.idx as id_of_n_apartments
        from buildings_filtered buildings
        left join zensus_apartment_data_3035 zensus_apartments
        on ST_DWithin(zensus_apartments.geometry, buildings.geometry, 77.7) -- radius is around 77.7 meters
    ) buildings_with_apartments
    where buildings_with_apartments.osm_id is not null

-- ^- this table will be droppped at a later point due to the need is only for preprocessing steps


drop table if exists buildings_with_res_temp

create table buildings_with_res_temp as
    select buildings_with_res_tmp.osm_id, buildings_with_res_tmp.amenity, buildings_with_res_tmp.building, buildings_with_res_tmp.name, buildings_with_res_tmp.geometry, buildings_with_res_tmp.area, buildings_with_res_tmp.tags, buildings_with_res_tmp.n_apartments, buildings_with_res_tmp.id_of_n_apartments, bwa.n_apartments_in_n_buildings
    from buildings_with_res_tmp
    left join (
        select * from (
            select b_res.id_of_n_apartments, count(b_res.*) as n_apartments_in_n_buildings from buildings_with_res_tmp b_res
            group by b_res.id_of_n_apartments
        ) bwa ) bwa
    on buildings_with_res_tmp.id_of_n_apartments = bwa.id_of_n_apartments


-- create index
CREATE INDEX ON buildings_with_res_temp USING gist (geometry);






drop table if exists amenities_in_buildings_tmp

CREATE TABLE amenities_in_buildings_tmp as
    with amenity as (select * from amenities_filtered af)
    select bf.osm_id as osm_id_building, bf.building, bf.area, bf.geometry as geometry_building, amenity.osm_id as osm_id_amenity, amenity.amenity, amenity.name, amenity.geometry as geometry_amenity, bf.tags as building_tags, amenity.tags as amenity_tags, bf.n_apartments, bf.id_of_n_apartments, bf.n_apartments_in_n_buildings
    from amenity, buildings_with_res_temp bf
    where st_intersects(bf.geometry, amenity.geometry)




drop table if exists buildings_with_amenities

CREATE TABLE buildings_with_amenities as
    select bwa.osm_id_amenity, bwa.osm_id_building, bwa.building, bwa.area, bwa.geometry_building, bwa.geometry_amenity,
    CASE
       WHEN (ST_Contains(bwa.geometry_building, ST_Centroid(bwa.geometry_building))) IS TRUE
       THEN ST_Centroid(bwa.geometry_building)
       ELSE ST_PointOnSurface(bwa.geometry_building)
    END AS geo_center,
    bwa."name", bwa.building_tags, bwa.amenity_tags, bwa.n_amenities_inside,
    case
        when n_apartments>0
        then bwa.n_apartments / bwa.n_apartments_in_n_buildings
        else 0
    end as n_apartments
    from (
        select bwa.osm_id_amenity, bwa.osm_id_building, bwa.building, bwa.area, bwa.geometry_building, bwa.geometry_amenity, bwa."name", bwa.building_tags, bwa.amenity_tags, bwa.n_amenities_inside, SUM(bwa.n_apartments) as n_apartments, SUM(bwa.n_apartments_in_n_buildings) as n_apartments_in_n_buildings
        from (
            select b.osm_id_amenity, b.osm_id_building, coalesce(b.amenity, b.building) as building, b.area, b.geometry_building, b.geometry_amenity, b.name, b.building_tags, b.amenity_tags, coalesce(b.n_apartments, 0) as n_apartments, coalesce(b.n_apartments_in_n_buildings, 0) as n_apartments_in_n_buildings, ainb.n_amenities_inside
            from amenities_in_buildings_tmp b
            left join (
                select ainb.osm_id_building, count(*) as n_amenities_inside from amenities_in_buildings_tmp ainb
                group by ainb.osm_id_building ) ainb
            on b.osm_id_building = ainb.osm_id_building
        ) bwa
        group by bwa.osm_id_amenity, bwa.osm_id_building, bwa.building, bwa.area, bwa.geometry_building, bwa.geometry_amenity, bwa."name", bwa.building_tags, bwa.amenity_tags, bwa.n_amenities_inside
    ) bwa



CREATE INDEX ON buildings_with_amenities USING gist (geometry_building);



drop table if exists buildings_without_amenities

-- get all buildings containing no amenities
CREATE TABLE buildings_without_amenities as
    select bwa.osm_id, bwa.building, bwa.area, bwa.geometry,
    CASE
       WHEN (ST_Contains(bwa.geometry, ST_Centroid(bwa.geometry))) IS TRUE
       THEN ST_Centroid(bwa.geometry)
       ELSE ST_PointOnSurface(bwa.geometry)
    END AS geo_center,
    bwa."name", bwa.tags, case when n_apartments>0 then bwa.n_apartments / bwa.n_apartments_in_n_buildings else 0 end as n_apartments
    from (
        select bwa.osm_id, bwa.building, bwa.area, bwa.geometry, bwa."name", bwa.tags, SUM(bwa.n_apartments) as n_apartments, SUM(bwa.n_apartments_in_n_buildings) as n_apartments_in_n_buildings
        from (
            select bf.osm_id, coalesce(bf.amenity, bf.building) as building, bf.name, bf.area, bf.geometry, bf.tags, coalesce(bf.n_apartments, 0) as n_apartments, coalesce(bf.n_apartments_in_n_buildings, 0) as n_apartments_in_n_buildings
            from buildings_with_res_temp bf
            where bf.osm_id not in (select aib.osm_id_building from amenities_in_buildings_tmp aib)
        ) bwa
        group by bwa.osm_id, bwa.building, bwa.area, bwa.geometry, bwa."name", bwa.tags
    ) bwa


CREATE INDEX ON buildings_without_amenities USING gist (geometry);




drop table if exists amenities_not_in_buildings

-- get all amenities not located in a building
CREATE TABLE amenities_not_in_buildings as
    select * from amenities_filtered af
    where af.osm_id not in (select aib.osm_id_amenity from amenities_in_buildings_tmp aib)

CREATE INDEX ON amenities_not_in_buildings USING gist (geometry);


