/**
 * 
 * THIS SCRIPT CREATES A TABLE WITH ZENSUS APARTMENT DATA FROM PREPROCESSED .CSV
 * AFTER RUNNING THIS SCRIPT, ZENSUS APARTMENT DATA CAN (AND WILL) BE MERGED WITH
 * BUILDINGS FROM OSM.
 *
 */


-- creaze temp without geometry
drop table if exists zensus_apartment_data_tmp

CREATE TABLE zensus_apartment_data_tmp
(i int primary key, x_mp int, y_mp int, apartments int, lon double precision, lat double precision)


copy zensus_apartment_data_tmp(i, x_mp, y_mp, apartments, lon, lat) FROM 'C:/Users/Public/Documents/buildings_preprocessed_100m.csv'
with (FORMAT csv, DELIMITER ',', header)



-- crate table with geometry
DROP TABLE if exists zensus_apartment_data

-- create table containing merged mandatory ways
CREATE TABLE zensus_apartment_data as
    select zad.i as idx, ST_SetSRID(ST_MakePoint(zad.lon, zad.lat), 4326) as geometry, zad.apartments as n_apartments
    from zensus_apartment_data_tmp zad


-- create index
CREATE INDEX ON zensus_apartment_data USING gist (geometry);


