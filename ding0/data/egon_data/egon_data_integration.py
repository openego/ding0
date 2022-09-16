# 1. VPN to RLI
# 2. Route db to port with (in command line/bash) with "ssh -NL ..."
# 3. check credentials 'PORT'
# 4. check imports, install additional modules? pip install saio, pip install pyyaml

import pandas as pd
import saio
from ding0.config.config_lv_grids_osm import get_config_osm
from geoalchemy2.shape import to_shape
from sqlalchemy import func, Integer, cast
from ding0.data.egon_data import egon_db

#create engine
egon_db.credentials()
engine = egon_db.engine()

# get schema metadata from db
saio.register_schema("demand", engine)
saio.register_schema("openstreetmap", engine)
saio.register_schema("boundaries", engine)

# get table metadata from db
# from saio.demand import tables of interest
from saio.demand import egon_household_electricity_profile_of_buildings, egon_building_electricity_peak_loads
from saio.openstreetmap import (osm_buildings_residential, osm_buildings_filtered_with_amenities,
                                osm_buildings_synthetic, osm_ways_with_segments, osm_buildings_filtered)
from saio.boundaries import egon_map_zensus_buildings_filtered, egon_map_zensus_buildings_filtered_all

def get_egon_residential_buildings(geo_area, scenario="eGon2035"):

    # retrieve residential buildings
    with egon_db.session_scope() as session:
        cells_query = session.query(
            osm_buildings_residential.id,
            osm_buildings_residential.osm_id,
            osm_buildings_residential.geom_building.label("geometry"),
            osm_buildings_residential.geom_point.label("raccordement_building"),
            osm_buildings_residential.building.label("category"),
            osm_buildings_residential.area,
            func.ST_X(osm_buildings_residential.geom_point).label("x"),
            func.ST_Y(osm_buildings_residential.geom_point).label("y"),
        ).filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                osm_buildings_residential.geom_point,
            )
        )

    df_osm_buildings_residential = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_residential["geometry"] = df_osm_buildings_residential[
        "geometry"
    ].apply(to_shape)
    df_osm_buildings_residential[
        "raccordement_building"
    ] = df_osm_buildings_residential["raccordement_building"].apply(to_shape)

    df_osm_buildings_residential["category"].replace("yes", "residential", inplace=True)

    # retrieve synthetic buildings
    with egon_db.session_scope() as session:
        cells_query = session.query(
            osm_buildings_synthetic.id,
            osm_buildings_synthetic.geom_building.label("geometry"),
            osm_buildings_synthetic.geom_point.label("raccordement_building"),
            osm_buildings_synthetic.building.label("category"),
            osm_buildings_synthetic.area,
            func.ST_X(osm_buildings_synthetic.geom_point).label("x"),
            func.ST_Y(osm_buildings_synthetic.geom_point).label("y"),
        ).filter(osm_buildings_synthetic.building == 'residential'
        ).filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                osm_buildings_synthetic.geom_point,
            )
        )

    df_osm_buildings_synthetic = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_synthetic["geometry"] = df_osm_buildings_synthetic[
        "geometry"
    ].apply(to_shape)
    df_osm_buildings_synthetic["raccordement_building"] = df_osm_buildings_synthetic[
        "raccordement_building"
    ].apply(to_shape)
    
    #TODO: id of synthetic buildings comes in str format -> convert to int in egon_db, workaround:
    df_osm_buildings_synthetic['id'] = df_osm_buildings_synthetic['id'].astype(int)

    # append residential and synthetic buildings
    df_all_buildings_residential = df_osm_buildings_residential.append(
        df_osm_buildings_synthetic, ignore_index=True
    )


    # retrieve number of households in buildings for selected buildings
    with egon_db.session_scope() as session:
        cells_query = (
            session.query(
                egon_household_electricity_profile_of_buildings.building_id.label("id"),
                func.count(
                    egon_household_electricity_profile_of_buildings.profile_id
                ).label("number_households"),
                # TODO: no status quo peak loads are available, grid are build based on 2035 scenario
                (egon_building_electricity_peak_loads.peak_load_in_w / 1000).label("capacity"),
            )
            .filter(
                egon_household_electricity_profile_of_buildings.building_id.in_(
                    df_all_buildings_residential['id'].values
                )
            )
            # TODO: check if .join() is slower than to create a third separate df with building peak loads
            .join(egon_building_electricity_peak_loads,
                  egon_household_electricity_profile_of_buildings.building_id ==
                  # TODO: id of building peak loads comes in str format -> convert to int in egon_db, workaround:
                  cast(egon_building_electricity_peak_loads.building_id, Integer)
                  ).filter(
                egon_building_electricity_peak_loads.sector == "residential"
            ).filter(
                # TODO: which scenario should be taken?
                egon_building_electricity_peak_loads.scenario == scenario
            ).group_by(
                egon_household_electricity_profile_of_buildings.building_id,
                egon_building_electricity_peak_loads.peak_load_in_w,
            )
        )

    df_apartments_per_building = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    df_all_buildings_residential = pd.merge(
        left=df_all_buildings_residential,
        right=df_apartments_per_building,
        left_on="id",
        right_on="id",
        how="left",
    )

    # TODO: do not import buildings with no_households or peak load == NaN, workaround:
    df_all_buildings_residential.dropna(axis=0, inplace=True)
    # set residential catgeory and egon_id to df index
    df_all_buildings_residential = df_all_buildings_residential.set_index('id')
    df_all_buildings_residential['category'] = 'residential'

    return df_all_buildings_residential


def get_egon_ways(geo_area):

    # retrieve ways
    with egon_db.session_scope() as session:
        cells_query = session.query(
            osm_ways_with_segments.osm_id,
            osm_ways_with_segments.nodes,
            osm_ways_with_segments.geom.label('geometry'),
            osm_ways_with_segments.highway,
            osm_ways_with_segments.length_segments,
        ).filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                osm_ways_with_segments.geom,
            )
        )

    # query needed for post processing  - return query instead of dataframe
    return cells_query

def get_egon_cts_buildings(geo_area, scenario="eGon2035"):

    # retrieve residential buildings
    with egon_db.session_scope() as session:
        cells_query = session.query(
            osm_buildings_filtered.id,
            osm_buildings_filtered.osm_id,
            osm_buildings_filtered.geom_building.label("geometry"),
            osm_buildings_filtered.geom_point.label("raccordement_building"),
            osm_buildings_filtered.building.label("category"),
            osm_buildings_filtered.area,
            func.ST_X(osm_buildings_filtered.geom_point).label("x"),
            func.ST_Y(osm_buildings_filtered.geom_point).label("y"),
        ).filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                osm_buildings_filtered.geom_point,
            )
        )

    df_osm_buildings_cts = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_cts["geometry"] = df_osm_buildings_cts[
        "geometry"
    ].apply(to_shape)
    df_osm_buildings_cts[
        "raccordement_building"
    ] = df_osm_buildings_cts["raccordement_building"].apply(to_shape)

    df_osm_buildings_cts["category"].replace("yes", "residential", inplace=True)

    # retrieve synthetic buildings
    with egon_db.session_scope() as session:
        cells_query = session.query(
            osm_buildings_synthetic.id,
            osm_buildings_synthetic.geom_building.label("geometry"),
            osm_buildings_synthetic.geom_point.label("raccordement_building"),
            osm_buildings_synthetic.building.label("category"),
            osm_buildings_synthetic.area,
            func.ST_X(osm_buildings_synthetic.geom_point).label("x"),
            func.ST_Y(osm_buildings_synthetic.geom_point).label("y"),
        ).filter(osm_buildings_synthetic.building == 'cts'
        ).filter(
            func.st_intersects(
                func.ST_GeomFromText(geo_area, get_config_osm("srid")),
                osm_buildings_synthetic.geom_point,
            )
        )

    df_osm_buildings_synthetic = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    # convert wkb-geom to shapely
    df_osm_buildings_synthetic["geometry"] = df_osm_buildings_synthetic[
        "geometry"
    ].apply(to_shape)
    df_osm_buildings_synthetic["raccordement_building"] = df_osm_buildings_synthetic[
        "raccordement_building"
    ].apply(to_shape)

    # TODO: id of synthetic buildings comes in str format -> convert to int in egon_db, workaround:
    df_osm_buildings_synthetic['id'] = df_osm_buildings_synthetic['id'].astype(int)

    # append residential and synthetic buildings
    df_all_buildings = df_osm_buildings_cts.append(
        df_osm_buildings_synthetic, ignore_index=True
    )

    # retrieve peak load of cts
    with egon_db.session_scope() as session:
        cells_query = session.query(
            egon_building_electricity_peak_loads.building_id.label("id"),
            egon_building_electricity_peak_loads.peak_load_in_w
        ).filter(
            egon_building_electricity_peak_loads.scenario == scenario
        ).filter(
            egon_building_electricity_peak_loads.sector == "cts"
        )

    df_cts_peak_load = pd.read_sql(
        sql=cells_query.statement, con=cells_query.session.bind, index_col=None
    )

    df_all_buildings_cts = pd.merge(
        left=df_all_buildings,
        right=df_cts_peak_load,
        left_on="id",
        right_on="id",
        how="left",
    )

    # TODO: do not import buildings with no_households or peak load == NaN, workaround:
    df_all_buildings_cts.dropna(axis=0, inplace=True)
    # set residential catgeory and egon_id to df index
    df_all_buildings_cts = df_all_buildings_cts.set_index('id')
    df_all_buildings_cts['category'] = 'cts'
    df_all_buildings_cts["number_households"] = 0

    return df_all_buildings_cts